#!/usr/bin/env python3
"""Validate and compare raw-feature-only baselines with the trained matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiments.setup.icl_arch_matrix.aggregate_results import (
    ARCHITECTURES,
    TARGETS,
)
from scripts.experiments.setup.icl_arch_matrix.evaluate_raw_features import BASELINES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--trained-reference", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.results)
    with input_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != len(BASELINES) * len(TARGETS):
        raise ValueError(f"expected eight raw-feature rows, got {len(rows)}")

    expected = {(baseline, target) for baseline in BASELINES for target in TARGETS}
    observed = [(row["baseline"], row["dataset"]) for row in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError(f"raw-feature grid mismatch: {observed}")
    for row in rows:
        if row["model_id"] != row["baseline"] or row["sources"] != []:
            raise ValueError(f"invalid raw-feature identity/source row: {row}")
        if int(row["seed"]) != 0 or int(row["training_updates"]) != 0:
            raise ValueError(f"invalid seed/update accounting: {row}")
        if row["feature_view"] != "l2_normalized_raw_768d_center":
            raise ValueError(f"unexpected feature view: {row}")
        if bool(row["topology_used"]):
            raise ValueError(f"raw-feature baseline used topology: {row}")
        for metric in ("roc_auc", "accuracy", "f1"):
            value = float(row[metric])
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid {metric}: {value}")

    trained = pd.read_csv(args.trained_reference)
    if len(trained) != 372:
        raise ValueError(f"expected 372 trained rows, got {len(trained)}")
    trained_fingerprints = trained.groupby("dataset").episode_fingerprint.unique()
    for target in TARGETS:
        fingerprints = {row["episode_fingerprint"] for row in rows if row["dataset"] == target}
        reference = trained_fingerprints.loc[target]
        if len(fingerprints) != 1 or len(reference) != 1:
            raise ValueError(f"nonunique fingerprint on {target}")
        if next(iter(fingerprints)) != reference[0]:
            raise ValueError(f"raw-feature fingerprint differs from trained matrix on {target}")

    rows.sort(key=lambda row: (BASELINES.index(row["baseline"]), TARGETS.index(row["dataset"])))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    fields = [
        "baseline", "model_id", "sources", "seed", "training_updates",
        "feature_view", "topology_used", "support_fit", "task", "dataset",
        "n_way", "n_shot", "n_query", "episodes", "queries",
        "episode_fingerprint", "roc_auc", "accuracy", "f1",
    ]
    with (output_root / "classification_long.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "sources": ""})

    raw_df = pd.DataFrame(rows)
    target_df = raw_df[["baseline", "dataset", "roc_auc", "accuracy", "f1"]].copy()
    target_df.to_csv(output_root / "target_summary.csv", index=False)

    comparisons = []
    for architecture in ARCHITECTURES:
        architecture_cells = trained[trained.architecture == architecture]
        for baseline in BASELINES:
            anchors = raw_df[raw_df.baseline == baseline].set_index("dataset").roc_auc
            deltas = architecture_cells.apply(
                lambda row: float(row.roc_auc - anchors.loc[row.dataset]), axis=1
            )
            comparisons.append(
                {
                    "architecture": architecture,
                    "baseline": baseline,
                    "baseline_mean_roc_auc": float(anchors.mean()),
                    "update100_mean_roc_auc": float(architecture_cells.roc_auc.mean()),
                    "mean_delta": float(deltas.mean()),
                    "update100_cells_above_baseline_fraction": float((deltas > 0).mean()),
                }
            )
    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(output_root / "architecture_comparison.csv", index=False)

    summary = {
        "protocol": "fixed_episode_raw_center_feature_control",
        "seed": 0,
        "rows": len(rows),
        "training_updates": 0,
        "feature_view": "l2_normalized_raw_768d_center",
        "topology_used": False,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "targets": list(TARGETS),
        "episode_fingerprints": {
            target: next(row["episode_fingerprint"] for row in rows if row["dataset"] == target)
            for target in TARGETS
        },
        "mean_roc_auc": {
            baseline: float(raw_df[raw_df.baseline == baseline].roc_auc.mean())
            for baseline in BASELINES
        },
        "architecture_comparison": {
            architecture: {
                baseline: {
                    "mean_delta": float(
                        comparison_df.loc[
                            (comparison_df.architecture == architecture)
                            & (comparison_df.baseline == baseline),
                            "mean_delta",
                        ].iloc[0]
                    ),
                    "cells_above_fraction": float(
                        comparison_df.loc[
                            (comparison_df.architecture == architecture)
                            & (comparison_df.baseline == baseline),
                            "update100_cells_above_baseline_fraction",
                        ].iloc[0]
                    ),
                }
                for baseline in BASELINES
            }
            for architecture in ARCHITECTURES
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
