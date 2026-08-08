#!/usr/bin/env python3
"""Import the strict final-core three-seed specialist transfer matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "data" / "final_core_matrix"

NAME_MAP = {
    "covid": "covid19_twitter",
    "ukr_rus": "ukr_rus_twitter",
    "cp_hk": "cp_hk_twitter",
    "midterm": "midterm",
    "twibot20": "twibot20",
    "election2020": "election2020",
    "covid_political": "covid_political",
    "ukr_rus_suspended": "ukr_rus_suspended",
    "facebook_page_reference": "facebook_page_reference",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-root", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    completeness = json.loads((args.summary_root / "completeness.json").read_text())
    expected = {
        "protocol": "fixed_test_512_static_test_on_static_train_v1",
        "checkpoint_step": 2500,
        "episode_count_per_cell": 512,
        "matrix_cells": 243,
    }
    for key, value in expected.items():
        if completeness.get(key) != value:
            raise ValueError(f"unexpected {key}: {completeness.get(key)!r}")

    raw = pd.read_csv(args.summary_root / "single_source_matrix_long.tsv", sep="\t")
    if len(raw) != 243 or raw[["seed", "model_id", "target"]].duplicated().any():
        raise ValueError("expected 243 unique three-seed specialist cells")
    if set(raw.seed) != {0, 1, 2}:
        raise ValueError("expected seeds 0, 1, and 2")
    if not raw.model_id.str.startswith("ss_").all():
        raise ValueError("matrix contains a non-specialist model")

    long = raw.rename(columns={"score": "accuracy"}).copy()
    long["source_key"] = long.model_id.str.removeprefix("ss_")
    if not set(long.source_key).issubset(NAME_MAP) or not set(long.target).issubset(NAME_MAP):
        raise ValueError("matrix contains an unknown graph name")
    long["source"] = long.source_key.map(NAME_MAP)
    long["target_key"] = long.target
    long["target"] = long.target.map(NAME_MAP)
    columns = [
        "seed", "source", "target", "source_key", "target_key", "accuracy",
        "score_std_across_batches", "loss", "checkpoint_step", "episode_count",
        "episode_plan_fingerprint", "observed_episode_fingerprint",
    ]
    long = long[columns].sort_values(["seed", "source", "target"]).reset_index(drop=True)

    graph_order = [NAME_MAP[key] for key in NAME_MAP]
    for seed, frame in long.groupby("seed"):
        if len(frame) != 81:
            raise ValueError(f"seed {seed} does not contain a complete 9x9 matrix")
        pivot = frame.pivot(index="source", columns="target", values="accuracy")
        if not np.isfinite(pivot.reindex(index=graph_order, columns=graph_order)).all().all():
            raise ValueError(f"seed {seed} matrix is incomplete after name alignment")

    summary = (
        long.groupby(["source", "target"], as_index=False)
        .accuracy.agg(["mean", "std"]).reset_index()
        .rename(columns={"mean": "accuracy_mean", "std": "accuracy_sample_std"})
    )
    canonical = summary.rename(columns={"source": "train", "target": "test", "accuracy_mean": "value"})
    canonical["metric"] = "accuracy"
    canonical = canonical[["train", "test", "metric", "value"]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.out_dir / "specialist_cells_three_seed.csv", index=False)
    summary.to_csv(args.out_dir / "specialist_cells_summary.csv", index=False)
    canonical.to_csv(args.out_dir / "transfer_matrix_three_seed_mean_long.csv", index=False)

    mean_matrix = summary.pivot(index="source", columns="target", values="accuracy_mean")
    std_matrix = summary.pivot(index="source", columns="target", values="accuracy_sample_std")
    mean_matrix.reindex(index=graph_order, columns=graph_order).to_csv(
        args.out_dir / "transfer_matrix_three_seed_mean.csv"
    )
    std_matrix.reindex(index=graph_order, columns=graph_order).to_csv(
        args.out_dir / "transfer_matrix_three_seed_sample_std.csv"
    )

    provenance = {
        **completeness,
        "source_summary_root": str(args.summary_root.resolve()),
        "imported_rows": len(long),
        "canonical_graph_names": graph_order,
        "outcome": "episodic 30-way neighbor-matching accuracy",
        "roc_auc_note": (
            "The fixed-grid result JSON and strict aggregate store score/accuracy. "
            "Trainer sidecars compute multiclass ROC-AUC, but ROC-AUC was not included "
            "in the complete 243-cell aggregate."
        ),
    }
    (args.out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
