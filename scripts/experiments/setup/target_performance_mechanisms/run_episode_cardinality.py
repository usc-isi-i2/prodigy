"""Replay the exact cached production suffix with prespecified count diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch

from experiments.layers import get_module_list
from experiments.run_shared_graph import write_json
from models.general_gnn import SingleLayerGeneralGNN
from scripts.experiments.setup.icl_arch_matrix.common_protocol import classification_targets
from scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy import resolved_params
from scripts.experiments.setup.final_core.core_plan import SOURCES
from .analyze_mixture_predictions import input_labels, evaluate_logits, METRICS
from .episode_cardinality import VARIANTS, cardinality_mode, cached_meta_forward
from .prepare_mixture_complementarity import TARGETS
from .replay import batch_hash
from .verify_member_training import model_digest


def make_model(protocol, target_name, graph_path, feature_dim, state):
    args = SimpleNamespace(**protocol)
    target = classification_targets(args.catalog, include_facebook=True)[target_name]
    params = resolved_params(args, target_name, target, Path(graph_path), None, "cached_cardinality", 2500)
    if params["layers"] != "S,U,M" or params["emb_dim"] != 256 or params["gnn_type"] != "sage":
        raise ValueError("unexpected historical model architecture")
    layers = get_module_list(params["layers"], params["emb_dim"], edge_attr_dim=None,
        input_dim=feature_dim, dropout=params["dropout"], reset_after_layer=params["reset_after_layer"],
        attention_mask_scheme=params["attention_mask_scheme"], has_final_back=params["has_final_back"],
        msg_pos_only=params["meta_gnn_pos_only"], batch_norm_metagraph=not params["no_bn_metagraph"],
        batch_norm_encoder=not params["no_bn_encoder"], encoder_gnn_type=params["gnn_type"])
    model = SingleLayerGeneralGNN(torch.nn.ModuleList(layers),
        initial_label_mlp=torch.nn.Linear(768, params["emb_dim"]), params=params,
        text_dropout=torch.nn.Dropout(params["text_features_dropout"]))
    model.load_state_dict(state, strict=True)
    return model.eval()


def choose_root(roots, target, ids):
    choices = []
    for root in roots:
        directory = root / target
        if not (directory / "metrics.jsonl").is_file():
            continue
        rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        full = [r for r in rows if r["variant"] == "baseline" and r["decoder"] == "full_model" and r["model_id"] in ids]
        if len(full) == len(ids) and {r["model_id"] for r in full} == ids and all((directory / f"{i}__baseline.pt").is_file() for i in ids):
            choices.append((root, {r["model_id"]: r for r in full}))
    if len(choices) != 1:
        raise ValueError(f"requires exactly one complete embedding cache for {target}, found {len(choices)}")
    return choices[0]


def validate_training(records):
    records = [r for r in records if r["model_id"].startswith("ss_")]
    if len(records) != 9 or {r["model_id"].removeprefix("ss_") for r in records} != set(SOURCES):
        raise ValueError("requires all nine historical singleton contracts")
    for row in records:
        expected = {"seed": 0, "n_way": 30, "n_shots": 3, "n_query": 4,
                    "batch_size": 4, "epochs": 1, "dataset_len_cap": 2500,
                    "task_name": "neighbor_matching", "neighbor_sampling_source_subset": row["model_id"].removeprefix("ss_")}
        if any(row["parameter_contract"].get(k) != v for k, v in expected.items()):
            raise ValueError("historical training cardinality/provenance differs")
        if hashlib.sha256(Path(row["config_path"]).read_bytes()).hexdigest() != row["config_sha256"]:
            raise ValueError("previously audited training configuration changed")
        if Path(row["checkpoint"]).name != "state_dict_2500.ckpt":
            raise ValueError("wrong comparison checkpoint")
    return {r["model_id"]: r for r in records}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--original-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--fresh-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--training-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.threads <= 8 or torch.cuda.is_available():
        raise ValueError("new output, bounded CPU threads, no visible GPU required")
    torch.set_num_threads(args.threads)
    training = validate_training(json.loads(args.training_audit.read_text()))
    choices = {(stream, target): choose_root(getattr(args, stream + "_roots"), target, set(training))
               for stream in ("original", "fresh") for target in sorted(TARGETS)}
    if args.dry_run:
        print("Nine specialists, five targets, two fixed streams, seven conditions: 630 full-model cells; no training/encoder passes.")
        return
    args.output.mkdir(parents=True)
    write_json(args.output / "protocol.json", {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "variants": list(VARIANTS), "training_audit": str(args.training_audit),
        "training_audit_sha256": hashlib.sha256(args.training_audit.read_bytes()).hexdigest(),
        "train_ways": 30, "train_shots": 3, "eval_ways": 2, "eval_shots": 10,
        "primary": "count_joint_attention vs baseline and inverse; positive mean foreign-donor AUC on Facebook and TwiBot, both streams",
        "query_outcomes_used_for_intervention": False, "new_training": False, "new_encoder_passes": 0,
        "new_distinct_negative_classes": False, "independent_training_seeds": 1,
        "original_roots": list(map(str, args.original_roots)), "fresh_roots": list(map(str, args.fresh_roots)),
        "threads": args.threads})
    rows, inventories, input_inventory, attention = [], [], [], []
    verified_batches = 0
    with torch.no_grad():
        for (stream, target), (root, full_rows) in choices.items():
            directory = root / target
            labels = input_labels(directory, target)
            protocol = json.loads((root / "protocol.json").read_text())
            if protocol["eval_episode_seed_offset"] != (0 if stream == "original" else 100003) or not protocol["save_embeddings"]:
                raise ValueError("wrong episode stream or missing embeddings")
            batches = []
            feature_dim = None
            for index in range(32):
                batch = torch.load(directory / "batches" / f"batch_{index:03d}.pt", map_location="cpu", weights_only=False)
                if batch_hash(batch) != labels["cache"]["batch_sha256"][index]:
                    raise ValueError("cached test input changed")
                feature_dim = batch[0].x.shape[1]
                batches.append([None] + list(batch[1:]))
            input_inventory.append({"stream": stream, "target": target, "cache_root": str(directory),
                "queries": len(labels["local_y"]), **labels["cache"]})
            for model_id, prior in sorted(full_rows.items()):
                contract = training[model_id]
                if prior["checkpoint"] != contract["checkpoint"] or prior["sources"] != [model_id.removeprefix("ss_")]:
                    raise ValueError("saved embedding model identity differs")
                state = torch.load(prior["checkpoint"], map_location="cpu", weights_only=True)["model"]
                digest = model_digest(state)
                if digest != prior["weights_sha256"]:
                    raise ValueError("checkpoint differs from saved embeddings")
                model = make_model(protocol, target, labels["cache"]["graph_path"], feature_dim, state)
                path = directory / f"{model_id}__baseline.pt"
                saved = torch.load(path, map_location="cpu", weights_only=False)
                if len(saved) != 32 or prior["episode_fingerprint"] != labels["cache"]["episode_fingerprint"]:
                    raise ValueError("incomplete/mismatched saved embedding cell")
                outputs = {variant: [] for variant in VARIANTS}
                audits = {variant: [] for variant in VARIANTS}
                for index, (record, batch) in enumerate(zip(saved, batches)):
                    if record["batch"] != index or record["batch_sha256"] != labels["cache"]["batch_sha256"][index]:
                        raise ValueError("embedding input/order differs")
                    pre = record["embeddings"]["U1_pre_meta"]
                    for variant in VARIANTS:
                        with cardinality_mode(model.layer_list[2], variant) as audit:
                            post, logits = cached_meta_forward(model, batch, pre)
                        if len(audit) != 1 or audit[0]["eval_ways"] != 2 or audit[0]["eval_shots"] != 10:
                            raise ValueError("actual metagraph count contract differs")
                        if variant == "baseline":
                            torch.testing.assert_close(post, record["embeddings"]["M2_post_meta"], rtol=0, atol=0)
                            torch.testing.assert_close(logits, record["logits"]["full_model"], rtol=0, atol=0)
                            verified_batches += 1
                        if not torch.isfinite(logits).all():
                            raise ValueError("nonfinite intervention logits")
                        outputs[variant].append(logits)
                        audits[variant].append(audit[0])
                    _, restored = cached_meta_forward(model, batch, pre)
                    torch.testing.assert_close(restored, record["logits"]["full_model"], rtol=0, atol=0)
                if model_digest(model.state_dict()) != digest:
                    raise ValueError("checkpoint tensors mutated")
                for variant in VARIANTS:
                    prediction = torch.cat(outputs[variant])
                    result = evaluate_logits(prediction, labels)
                    if variant == "baseline" and max(abs(result[k] - prior[k]) for k in METRICS) > 1e-6:
                        raise ValueError("baseline production metric parity failed")
                    rows.append({"stream": stream, "target": target, "model_id": model_id,
                        "source": prior["sources"][0], "variant": variant,
                        "checkpoint": prior["checkpoint"], "weights_sha256": digest,
                        "episode_fingerprint": prior["episode_fingerprint"], "episodes": 128,
                        "queries": len(labels["local_y"]), **result})
                    attention.append({"stream": stream, "target": target, "model_id": model_id,
                        "variant": variant, "batches": audits[variant]})
                destination = args.output / "predictions" / stream / target
                destination.mkdir(parents=True, exist_ok=True)
                artifact = destination / f"{model_id}.pt"
                torch.save({"logits": {v: torch.cat(outputs[v]) for v in VARIANTS},
                    "batch_sha256": labels["cache"]["batch_sha256"], "weights_sha256": digest}, artifact)
                inventories.append({"stream": stream, "target": target, "model_id": model_id,
                    "checkpoint": prior["checkpoint"], "weights_sha256": digest,
                    "baseline_suffix_bit_exact_batches": 32,
                    "embedding_file": str(path), "embedding_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "prediction_file": str(artifact), "prediction_file_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()})
                print(json.dumps({"stream": stream, "target": target, "model_id": model_id,
                    "completed_cells": len(rows), "baseline_bit_exact": True}), flush=True)
            write_json(args.output / "metrics.json", rows)
            write_json(args.output / "inventory.json", inventories)
            write_json(args.output / "input_inventory.json", input_inventory)
            write_json(args.output / "attention.json", attention)
    if len(rows) != 630 or len(inventories) != 90 or verified_batches != 2880:
        raise ValueError("incomplete prospective grid")
    write_json(args.output / "DONE.json", {"full_model_cells": len(rows), "model_target_stream_cells": len(inventories),
        "baseline_suffix_bit_exact_batches": verified_batches, "all_weights_unchanged": True,
        "all_cached_inputs_verified": True, "all_baseline_metrics_reproduced": True})


if __name__ == "__main__":
    main()
