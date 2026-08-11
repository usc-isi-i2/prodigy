#!/usr/bin/env python3
"""Validate and aggregate the descriptive one-seed CLS architecture matrix."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from scripts.experiments.setup.final_core.core_plan import build_models


ARCHITECTURES = ("prodigy", "vision", "gilt")
TARGETS = ("covid_political", "election2020", "ukr_rus_suspended", "twibot20")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prodigy", required=True)
    parser.add_argument("--vision", required=True)
    parser.add_argument("--gilt", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model-ids", default="")
    return parser.parse_args()


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    args = parse_args()
    paths = {architecture: getattr(args, architecture) for architecture in ARCHITECTURES}
    rows = []
    for architecture, path in paths.items():
        loaded = read_jsonl(path)
        if any(row["architecture"] != architecture for row in loaded):
            raise ValueError(f"architecture mismatch in {path}")
        rows.extend(loaded)

    selected = set(filter(None, args.model_ids.split(",")))
    plan_models = [model for model in build_models() if not selected or model.model_id in selected]
    if selected != {model.model_id for model in plan_models} and selected:
        raise ValueError(f"unknown model ids: {sorted(selected - {m.model_id for m in plan_models})}")
    model_order = {model.model_id: index for index, model in enumerate(plan_models)}
    expected = {
        (architecture, model_id, target)
        for architecture in ARCHITECTURES
        for model_id in model_order
        for target in TARGETS
    }
    observed = [(row["architecture"], row["model_id"], row["dataset"]) for row in rows]
    if len(observed) != len(set(observed)):
        raise ValueError("duplicate architecture/model/target result rows")
    missing, extra = expected - set(observed), set(observed) - expected
    if missing or extra:
        raise ValueError(f"result grid mismatch: missing={sorted(missing)} extra={sorted(extra)}")

    for target in TARGETS:
        fingerprints = {row["episode_fingerprint"] for row in rows if row["dataset"] == target}
        if len(fingerprints) != 1:
            raise ValueError(f"architectures did not consume identical {target} episodes")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    long_fields = [
        "architecture", "model_id", "sources", "seed", "checkpoint_step", "task",
        "dataset", "n_way", "n_shot", "n_query", "episodes", "queries",
        "episode_fingerprint", "roc_auc", "accuracy", "f1",
    ]
    rows.sort(key=lambda row: (
        ARCHITECTURES.index(row["architecture"]),
        model_order[row["model_id"]],
        TARGETS.index(row["dataset"]),
    ))
    with (output_root / "classification_long.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=long_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "sources": ",".join(row["sources"])})

    lookup = {(r["architecture"], r["model_id"], r["dataset"]): r for r in rows}
    wide_fields = ["architecture", "model_id", "n_sources", "sources", *TARGETS, "mean_roc_auc"]
    wide_rows = []
    for architecture in ARCHITECTURES:
        for plan_model in plan_models:
            scores = [lookup[(architecture, plan_model.model_id, target)]["roc_auc"] for target in TARGETS]
            wide_rows.append({
                "architecture": architecture,
                "model_id": plan_model.model_id,
                "n_sources": len(plan_model.sources),
                "sources": ",".join(plan_model.sources),
                **dict(zip(TARGETS, scores)),
                "mean_roc_auc": sum(scores) / len(scores),
            })
    with (output_root / "classification_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=wide_fields)
        writer.writeheader()
        writer.writerows(wide_rows)

    architecture_scores = defaultdict(list)
    for row in rows:
        architecture_scores[row["architecture"]].append(float(row["roc_auc"]))
    summary = {
        "protocol": "descriptive_one_seed_fixed_episode_classification_comparison",
        "seed": 0,
        "checkpoint_step": 500,
        "models_per_architecture": len(model_order),
        "targets": list(TARGETS),
        "mean_roc_auc_over_all_cells": {
            architecture: sum(scores) / len(scores)
            for architecture, scores in architecture_scores.items()
        },
        "episode_fingerprints": {
            target: next(row["episode_fingerprint"] for row in rows if row["dataset"] == target)
            for target in TARGETS
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
