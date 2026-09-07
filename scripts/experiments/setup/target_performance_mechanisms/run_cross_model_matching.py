"""Compare frozen model families on identical two-way target episodes and readouts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from .analyze_mixture_predictions import input_labels, evaluate_logits
from .replay import batch_hash, episode_probe
from .prepare_mixture_complementarity import TARGETS


def matched_prototypes(embeddings, batch, graph_labels):
    """Use only support labels; query labels are checked outside prediction."""
    g = batch[0]
    centers = g.global_node_ids[g.ptr[:-1]]
    if (centers < 0).any() or centers.max() >= len(embeddings):
        raise ValueError("invalid target center IDs")
    local = batch[2].argmax(1)
    expected = g.task_label_map[g.task_id_per_sample, local]
    if not torch.equal(graph_labels[centers], expected):
        raise ValueError("cached and native labels disagree")
    return episode_probe(embeddings[centers], batch, "prototype")


def disagreement(a, b, truth):
    ac, bc = a.argmax(1) == truth, b.argmax(1) == truth
    return {"queries": len(truth), "both_correct": int((ac & bc).sum()),
        "both_wrong": int((~ac & ~bc).sum()), "prodigy_only_correct": int((ac & ~bc).sum()),
        "samgpt_only_correct": int((~ac & bc).sum())}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--samgpt-root", type=Path, required=True)
    p.add_argument("--plans", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--embeddings", type=Path, help="reuse a completed native export")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.output.exists() or torch.cuda.is_available():
        raise ValueError("new CPU-only output required")
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True)
    export = args.embeddings or args.output / "native_embeddings"
    if args.embeddings is None:
        command = [sys.executable, str(Path(__file__).with_name("export_samgpt_for_matching.py")),
            "--samgpt-root", str(args.samgpt_root), "--plans", str(args.plans),
            "--output", str(export), "--threads", str(args.threads)]
        subprocess.run(command + (["--smoke"] if args.smoke else []), check=True)
    done = json.loads((export / "DONE.json").read_text())
    if not done["complete"] or done["smoke"] != args.smoke:
        raise ValueError("matching-scope native embeddings required")
    plans = json.loads(args.plans.read_text())
    if args.smoke:
        plans = [r for r in plans if r["target"] == "covid_political" and r["stream"] == "original"]
    receipts = json.loads((export / "receipts.json").read_text())
    sources = sorted({r["source"] for r in receipts})
    if not args.smoke and (len(sources) != 9 or len(plans) != 10 or {r["target"] for r in plans} != TARGETS):
        raise ValueError("complete nine-source, five-target, two-stream design required")
    rows, errors, ranks, cases = [], [], [], []
    data = Path(__file__).parents[2] / "analysis/graphs/transfer_prediction/target_performance_mechanisms/data"
    examples = json.loads((data / "individual_examples.json").read_text())
    for plan in plans:
        target, stream = plan["target"], plan["stream"]
        root = Path(plan["original"]["root"])
        labels = input_labels(root, target)
        native_inputs = torch.load(export / target / "inputs.pt", map_location="cpu", weights_only=False)
        if native_inputs["graph_file_sha256"] != plan["graph_file_sha256"]:
            raise ValueError("native graph changed")
        batches = [torch.load(root / "batches" / f"batch_{i:03d}.pt", map_location="cpu", weights_only=False) for i in range(32)]
        if any(batch_hash(b) != labels["cache"]["batch_sha256"][i] for i, b in enumerate(batches)):
            raise ValueError("cached inputs changed")
        predictions = {}
        for source in sources:
            native = torch.load(export / target / f"{source}.pt", map_location="cpu", weights_only=False)
            sam = torch.cat([matched_prototypes(native["embeddings"], b, native_inputs["labels"]) for b in batches])
            prod = torch.load(root / f"ss_{source}__baseline.pt", map_location="cpu", weights_only=False)
            if len(prod) != 32 or any(r["batch_sha256"] != labels["cache"]["batch_sha256"][i] for i, r in enumerate(prod)):
                raise ValueError("PRODIGY predictions are not on the same inputs")
            predictions[(source, "samgpt_prototype")] = sam
            for old, new in (("full_model", "prodigy_full"), ("U1_pre_meta/prototype", "prodigy_prototype")):
                logits = torch.cat([r["logits"][old] for r in prod])
                predictions[(source, new)] = logits
                errors.append({"target": target, "stream": stream, "source": source, "prodigy_readout": new,
                    **disagreement(logits, sam, labels["local_y"])})
            del native, prod
        projected = torch.cat([matched_prototypes(native_inputs["projected_features"], b, native_inputs["labels"]) for b in batches])
        raw = torch.cat([episode_probe(b[0].x[b[0].ptr[:-1]], b, "prototype") for b in batches])
        predictions[("none", "raw_768_prototype")] = raw
        predictions[("none", "raw_50_prototype")] = projected
        for (source, decoder), logits in predictions.items():
            rows.append({"target": target, "stream": stream, "source": source, "decoder": decoder,
                "episode_fingerprint": labels["cache"]["episode_fingerprint"], **evaluate_logits(logits, labels)})
        dest = args.output / "predictions" / stream
        dest.mkdir(parents=True, exist_ok=True)
        query_ids = torch.cat([b[0].global_node_ids[b[0].ptr[:-1]][b[5].reshape(-1, 2)[:, 0].bool()] for b in batches])
        torch.save({"predictions": predictions, "labels": labels, "query_node_ids": query_ids}, dest / f"{target}.pt")
        for decoder in ("prodigy_full", "prodigy_prototype"):
            part = pd.DataFrame([r for r in rows if r["target"] == target and r["stream"] == stream and r["source"] != "none"])
            table = part.pivot(index="source", columns="decoder", values="roc_auc")
            ranks.append({"target": target, "stream": stream, "prodigy_readout": decoder,
                "source_rank_spearman": float(table[decoder].corr(table.samgpt_prototype, method="spearman"))})
        if stream == "original":
            for group in examples:
                if group["target"] != target:
                    continue
                for case in group["cases"]:
                    ix = sum(labels["batch_counts"][:case["batch"]]) + case["query_index"]
                    if str(int(query_ids[ix])) != case["node"]:
                        raise ValueError("individual example identity mismatch")
                    for (source, decoder), logits in predictions.items():
                        true = int(labels["local_y"][ix])
                        cases.append({"case_id": case["case_id"], "target": target, "node": case["node"],
                            "source": source, "decoder": decoder, "correct": bool(logits[ix].argmax() == true),
                            "p_label": float(logits[ix].softmax(0)[true]), "margin": float(logits[ix, true] - logits[ix, 1-true])})
        pd.DataFrame(rows).to_csv(args.output / "cells.csv", index=False)
        pd.DataFrame(errors).to_csv(args.output / "error_agreement.csv", index=False)
        pd.DataFrame(ranks).to_csv(args.output / "source_ranks.csv", index=False)
        pd.DataFrame(cases).to_csv(args.output / "individual_examples.csv", index=False)
        print(f"Matched {stream}/{target}: {len(predictions)} prediction sets", flush=True)
    (args.output / "protocol.json").write_text(json.dumps({"smoke": args.smoke,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "all_target_support_query_ids_and_labels_matched": True, "ways": 2, "shots_per_class": 10,
        "episodes_per_target_stream": 128, "shared_readout": "normalize nodes, average support by class, cosine",
        "metric": "pooled query AUC with canonical global/local class mapping",
        "samgpt_graph_context": "native full symmetrized target graph; three GCN layers",
        "prodigy_graph_context": "existing sampled directed subgraphs with trained S/U pooling",
        "training": "existing SAMGPT GraphCL seed39 step500 vs PRODIGY NM seed0 step2500",
        "source_graph_views_matched": False, "pure_architecture_causal_effect": False,
        "calibration_temperatures_matched_for_full_prodigy": False,
        "new_training": False, "episode_streams_are_not_training_seed_replicas": True}, indent=2) + "\n")
    (args.output / "DONE.json").write_text(json.dumps({"complete": True, "smoke": args.smoke,
        "metric_cells": len(rows), "source_rank_comparisons": len(ranks), "prediction_cohorts": len(errors),
        "native_embedding_cells": done["model_target_cells"]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
