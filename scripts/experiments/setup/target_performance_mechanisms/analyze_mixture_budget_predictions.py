"""Reuse saved prediction tensors for fixed specialist-training-budget comparisons."""
import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd
import torch

from experiments.run_shared_graph import write_json
from .analyze_mixture_predictions import input_labels, load_predictions, compare_predictions
from scripts.experiments.analysis.graphs.transfer_prediction.target_performance_mechanisms.analyze_mixture_budget import (
    STEPS, TARGETS, STREAMS, budget_step, model_registry, validate_artifacts, validate_budget)


def main():
    repo = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=repo / "scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/data")
    parser.add_argument("--trajectory-root", type=Path, default=Path("/dataMeR1/phil/gfm/prodigy-mechanisms/log/target_mechanisms"))
    parser.add_argument("--mixture-replay", type=Path, default=Path("/dataMeR1/phil/gfm/prodigy-mechanisms-complement/log/target_mechanisms/mixture_complementarity_20260906_v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.threads <= 8 or torch.cuda.is_available():
        raise ValueError("existing output, invalid threads, or visible GPU")
    torch.set_num_threads(args.threads)
    prior = validate_artifacts(args.data / "mixture_complementarity_predictions", args.data / "mixture_complementarity_inputs")
    manifest = pd.read_csv(repo / "scripts/experiments/setup/target_performance_mechanisms/data/trajectory_model_list.tsv", sep="\t")
    models = model_registry(manifest, pd.read_json(args.data / "trajectory_checkpoint_inventory.json"), prior)
    for stream in STREAMS:
        directory = args.trajectory_root / f"trajectory_{stream}_20260906"
        if not (directory / "DONE").is_file():
            raise ValueError("complete saved trajectory required")
        protocol = json.loads((directory / "protocol.json").read_text())
        if any(protocol.get(k) != v for k, v in {"batch_count": 32, "training_seed": 0, "variants": "baseline",
                "eval_episode_seed_offset": 0 if stream == "original" else 100003}.items()):
            raise ValueError("historical trajectory protocol differs")
        for target in TARGETS:
            for file in (directory / target / "metrics.jsonl", args.mixture_replay / stream / target / "metrics.jsonl"):
                if not file.is_file():
                    raise FileNotFoundError(file)
    if args.dry_run:
        print(json.dumps({"models": 81, "prediction_cells": 810, "comparisons": 1800, "error_strata": 5400,
                          "pair_selected_step": budget_step(2), "loo_selected_step": budget_step(8), "new_training": False}))
        return
    args.output.mkdir(parents=True)
    tables = {name: [] for name in ("model_metrics", "prediction_inventory", "comparisons", "error_strata", "input_inventory")}
    records = models.to_dict("records")
    specialists = {(m["sources"][0], m["step"]): m["model_id"] for m in records if m["source_count"] == 1}
    for stream in STREAMS:
        for target in sorted(TARGETS):
            single_dir = args.trajectory_root / f"trajectory_{stream}_20260906" / target
            mixture_dir = args.mixture_replay / stream / target
            labels = input_labels(single_dir, target)
            expected = prior["input_inventory"].query("stream == @stream and target == @target").iloc[0]
            for field in ("batch_sha256", "episode_fingerprint", "episodes", "graph_path"):
                if labels["cache"][field] != expected[field]:
                    raise ValueError("trajectory test inputs differ from mixture comparison")
            tables["input_inventory"].append({"stream": stream, "target": target,
                "production_metric_uses_global_binary_labels": labels["use_global"], "queries": len(labels["local_y"]), **labels["cache"]})
            rows = {str(d): [json.loads(line) for line in (d / "metrics.jsonl").read_text().splitlines()] for d in (single_dir, mixture_dir)}
            predictions, scores = {}, {}
            for model in records:
                directory = single_dir if model["source_count"] == 1 else mixture_dir
                value, metrics, audit = load_predictions(directory, model, labels, rows[str(directory)])
                predictions[model["model_id"]], scores[model["model_id"]] = value, metrics
                tables["model_metrics"].append({"stream": stream, "target": target,
                    **{k: model[k] for k in ("model_id", "sources", "source_count", "step")}, **metrics})
                tables["prediction_inventory"].append({"stream": stream, "target": target, **audit})
            for model in (m for m in records if m["source_count"] > 1):
                for step in STEPS:
                    members = [specialists[(s, step)] for s in model["sources"]]
                    common = {"stream": stream, "target": target, "model_id": model["model_id"],
                        "sources": model["sources"], "source_count": model["source_count"], "target_seen": target in model["sources"],
                        "queries": len(labels["local_y"]), "specialist_step": step, "total_specialist_updates": len(members) * step,
                        "specialist_training_episodes": 4 * len(members) * step, "mixture_updates": 2500,
                        "mixture_training_episodes": 10000, "inference_model_multiplier": len(members),
                        "budget_selected": step == budget_step(len(members))}
                    row, strata = compare_predictions(common, torch.stack([predictions[m] for m in members]),
                        [scores[m] for m in members], predictions[model["model_id"]], scores[model["model_id"]], labels)
                    tables["comparisons"].append(row)
                    tables["error_strata"].extend(strata)
            print(json.dumps({"stream": stream, "target": target, "verified_predictions": 81, "budget_comparisons": 180}), flush=True)
    validate_budget({name: pd.DataFrame(values) for name, values in tables.items()}, models, prior)
    for name, values in tables.items():
        write_json(args.output / f"{name}.json", values)
    write_json(args.output / "protocol.json", {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "steps": list(STEPS), "training_seeds": [0], "mixture_step": 2500, "new_training": False,
        "ensemble_weights_fitted": False, "matched_training_inputs": False, "matched_flops": False,
        "causal_interference_claim": False, "budget_rule": "largest saved specialist step with source_count * step <= 2500",
        "inference_models": [2, 8], "primary_ensemble": "equal probability", "secondary_ensemble": "equal logits",
        "historical_sampler_only": True, "mixture_replay": str(args.mixture_replay), "trajectory_root": str(args.trajectory_root)})
    write_json(args.output / "DONE.json", {"models": 81, "prediction_cells": 810, "comparisons": 1800, "error_strata": 5400,
        "complete_both_streams": True, "terminal_reference_reproduced": True})


if __name__ == "__main__":
    main()
