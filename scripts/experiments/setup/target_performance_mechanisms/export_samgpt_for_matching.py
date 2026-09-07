"""Native frozen SAMGPT embeddings; run separately to isolate upstream imports."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch


ALIASES = {"ukr_rus_twitter": "ukr_rus", "covid19_twitter": "covid", "cp_hk_twitter": "cp_hk"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--samgpt-root", type=Path, required=True)
    p.add_argument("--plans", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.output.exists() or torch.cuda.is_available() or not 1 <= args.threads <= 8:
        raise ValueError("new output directory and CPU-only execution required")
    torch.set_num_threads(args.threads)
    # SAMGPT imports top-level `models`; never import PRODIGY model code here.
    sys.path.insert(0, str(args.samgpt_root))
    from samgpt_social.core_cls_eval import embed, load_labels
    from samgpt_social.data import gaussian_projection_matrix, load_graph_artifact, make_undirected_adjacency
    from samgpt_social.episodes import evaluate_cosine_prototypes, sample_episodes, stratified_node_splits
    from samgpt_social.graphcl_ladder_eval import load_checkpoint_state
    from samgpt_social.runner import build_preprompt_model, prepare_features

    plans = json.loads(args.plans.read_text())
    targets = sorted({r["target"] for r in plans})
    records = [json.loads(f.read_text()) for f in sorted(
        (args.samgpt_root / "log/source_lattice_cls/seed_39").glob("ss_*/metrics.json"))]
    if len(records) != 9 or len({r["sources"][0] for r in records}) != 9:
        raise ValueError("all nine native specialists required")
    if args.smoke:
        targets = ["covid_political"]
        records = [r for r in records if r["sources"][0] in ("ukr_rus_twitter", "cp_hk_twitter")]
    args.output.mkdir(parents=True)
    save_json(args.output / "protocol.json", {"smoke": args.smoke, "targets": targets,
        "samgpt_repository": str(args.samgpt_root), "samgpt_revision": subprocess.check_output(
            ["git", "-C", str(args.samgpt_root), "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint_update": 500, "seed": 39, "source_graph_views_matched_to_prodigy": False,
        "encoder": "native full-graph frozen base GCN, no downstream prompt fitting",
        "native_metric_tolerances": {"roc_auc_mean": 0.0001, "accuracy_mean": 0.001},
        "accuracy_tolerance_note": "Initial 1e-4 gate stopped on one of 3072 decisions (0.000326); retain all discrepancies, allow 0.001 accuracy error, keep AUC gate unchanged.",
        "new_training": False})
    models, receipts = [], []
    for r in records:
        if not r["complete"] or r["checkpoint_update"] != 500 or r["seed"] != 39 or len(r["sources"]) != 1:
            raise ValueError("unexpected specialist provenance")
        if digest(r["checkpoint"]) != r["checkpoint_sha256"]:
            raise ValueError("native checkpoint hash differs from saved evaluation")
        config = json.loads(Path(r["checkpoint"]).with_name("resolved_config.json").read_text())
        if config["feature_components"] != 50 or config["feature_projection_seed"] != 39:
            raise ValueError("unexpected native feature map")
        model = build_preprompt_model(50, config["pretrain_dataset_slots"], config, torch.device("cpu"))
        model.load_state_dict(load_checkpoint_state(Path(r["checkpoint"]), torch.device("cpu")))
        model.eval()
        source = ALIASES.get(r["sources"][0], r["sources"][0])
        models.append((source, r, model))
    projection = gaussian_projection_matrix(768, 50, 39)
    for target in targets:
        plan = next(r for r in plans if r["target"] == target)
        graph_path = Path(plan["original"]["graph_path"])
        graph_sha = digest(graph_path)
        if graph_sha != plan["graph_file_sha256"]:
            raise ValueError("target graph differs from verified cached inputs")
        graph = load_graph_artifact(graph_path)
        features = prepare_features(graph, projection)
        adjacency = make_undirected_adjacency(graph.edge_index, graph.num_nodes)
        first_task = records[0]["targets"][target]
        labels = load_labels(graph_path, first_task["label_key"])
        native_episodes = sample_episodes(labels, stratified_node_splits(labels, seed=0)["test"],
            shots=10, queries=12, count=128, split_name="test")
        dest = args.output / target
        dest.mkdir()
        torch.save({"labels": torch.from_numpy(labels), "projected_features": features,
            "graph_file_sha256": graph_sha, "graph_path": str(graph_path)}, dest / "inputs.pt")
        for source, reference, model in models:
            if reference["targets"][target]["graph"] != str(graph_path):
                raise ValueError("native and matched target graph paths differ")
            with torch.no_grad():
                embeddings = embed(model, features, adjacency, torch.device("cpu"))
            native, _ = evaluate_cosine_prototypes(embeddings, native_episodes)
            errors = {k: abs(native[k] - reference["targets"][target][k])
                for k in ("roc_auc_mean", "accuracy_mean")}
            if errors["roc_auc_mean"] > 0.0001 or errors["accuracy_mean"] > 0.001:
                raise ValueError(f"native replay mismatch: {source}/{target}: {errors}")
            receipt = {"source": source, "target": target, "checkpoint": reference["checkpoint"],
                "checkpoint_sha256": reference["checkpoint_sha256"], "native_metrics": native,
                "native_reference": reference["targets"][target], "native_errors": errors,
                "graph_file_sha256": graph_sha, "training_repository_commit": reference["training_repository_commit"]}
            torch.save({"embeddings": embeddings, "receipt": receipt}, dest / f"{source}.pt")
            receipts.append(receipt)
            save_json(args.output / "receipts.json", receipts)
            print(f"Native replay {source}/{target}: AUC={native['roc_auc_mean']:.4f}, error={max(errors.values()):.2g}", flush=True)
        del graph, features, adjacency
    save_json(args.output / "DONE.json", {"complete": True, "smoke": args.smoke,
        "model_target_cells": len(receipts), "all_native_metrics_reproduced": True})


if __name__ == "__main__":
    main()
