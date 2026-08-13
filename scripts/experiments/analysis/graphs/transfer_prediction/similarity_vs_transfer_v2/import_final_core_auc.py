#!/usr/bin/env python3
"""Import the proper three-seed final-core accuracy/F1/ROC-AUC matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_SUMMARY = HERE / "data" / "final_core_auc" / "raw"
DEFAULT_OUT = HERE / "data" / "final_core_auc"
METRICS = ("accuracy", "f1_macro", "roc_auc_ovr_macro")
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
    parser.add_argument("--summary-root", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    completeness = json.loads((args.summary_root / "completeness.json").read_text())
    expected = {
        "protocol": "fixed_test_512_static_test_on_static_train_v1",
        "metric_contract": "accuracy_f1_macro_roc_auc_ovr_macro_v1",
        "checkpoint_step": 2500,
        "episode_count_per_cell": 512,
        "specialist_cells": 243,
        "training_seeds": [0, 1, 2],
    }
    for key, value in expected.items():
        if completeness.get(key) != value:
            raise ValueError(f"unexpected {key}: {completeness.get(key)!r}")
    if tuple(completeness.get("metrics", ())) != METRICS:
        raise ValueError(f"unexpected metrics: {completeness.get('metrics')!r}")

    raw = pd.read_csv(args.summary_root / "single_source_metrics_long.tsv", sep="\t")
    key = ["seed", "source", "target"]
    if len(raw) != 243 or raw[key].duplicated().any():
        raise ValueError("expected 243 unique source/target/seed cells")
    if set(raw.seed) != {0, 1, 2}:
        raise ValueError("expected seeds 0, 1, and 2")
    if set(raw.source) != set(NAME_MAP) or set(raw.target) != set(NAME_MAP):
        raise ValueError("matrix graph names do not match the final-core catalog")
    if not np.isfinite(raw[list(METRICS)].to_numpy(float)).all():
        raise ValueError("matrix contains a missing or non-finite metric")

    cells = raw.rename(columns={"source": "source_key", "target": "target_key"}).copy()
    cells["source"] = cells.source_key.map(NAME_MAP)
    cells["target"] = cells.target_key.map(NAME_MAP)
    columns = [
        "seed", "source", "target", "source_key", "target_key", *METRICS,
        "checkpoint_step", "episode_count", "episode_plan_fingerprint",
        "observed_episode_fingerprint",
    ]
    cells = cells[columns].sort_values(["seed", "source", "target"]).reset_index(drop=True)
    graph_order = [NAME_MAP[key] for key in NAME_MAP]
    for seed, frame in cells.groupby("seed"):
        if len(frame) != 81:
            raise ValueError(f"seed {seed} does not contain a complete 9x9 matrix")
        for metric in METRICS:
            pivot = frame.pivot(index="source", columns="target", values=metric)
            aligned = pivot.reindex(index=graph_order, columns=graph_order)
            if not np.isfinite(aligned.to_numpy(float)).all():
                raise ValueError(f"seed {seed} {metric} matrix is incomplete")

    grouped = cells.groupby(["source", "target"])
    summary = grouped[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    long_rows = []
    for metric in METRICS:
        frame = summary[["source", "target", f"{metric}_mean"]].rename(
            columns={"source": "train", "target": "test", f"{metric}_mean": "value"}
        )
        frame["metric"] = metric
        long_rows.append(frame[["train", "test", "metric", "value"]])
    canonical = pd.concat(long_rows, ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.out_dir / "specialist_cells_three_seed.csv", index=False)
    summary.to_csv(args.out_dir / "specialist_cells_summary.csv", index=False)
    canonical.to_csv(args.out_dir / "transfer_matrix_three_seed_mean_long.csv", index=False)
    for metric in METRICS:
        for stat, suffix in (("mean", "mean"), ("std", "sample_std")):
            matrix = summary.pivot(
                index="source", columns="target", values=f"{metric}_{stat}"
            ).reindex(index=graph_order, columns=graph_order)
            matrix.to_csv(args.out_dir / f"transfer_matrix_{metric}_three_seed_{suffix}.csv")

    provenance = {
        **completeness,
        "source_summary_root": str(args.summary_root),
        "imported_rows": len(cells),
        "canonical_graph_names": graph_order,
        "outcomes": list(METRICS),
        "auc_definition": "multiclass one-vs-rest macro ROC-AUC from all episode logits",
        "diagonal_policy": "preserved in evidence; excluded from transfer-predictor analyses",
    }
    (args.out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
