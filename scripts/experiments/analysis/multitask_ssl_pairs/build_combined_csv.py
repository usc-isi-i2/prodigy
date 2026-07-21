#!/usr/bin/env python3
"""Merge the three per-task benchmark CSVs into ONE tidy long CSV covering all 7
arms of the {nm,cl,fp} subset lattice (single / pair / 3-way), for analysis.

Reads (keyed by ``model`` = arm):
    <plotting-root>/node_classification/data/node_classification.csv
    <plotting-root>/node_regression/data/node_regression.csv
    <plotting-root>/static_link_prediction/data/static_link_prediction.csv

Writes one row per (arm, task, dataset, target, split, shots), with the union of
all metric columns (blank where a metric does not apply to that task) plus derived
``group`` (single/pair/triple) and ``k`` (# objectives in the rotation) columns:

    group,k,model,task,dataset,target,split,shots,roc_auc,accuracy,f1,spearman,rmse,mae,r2,mse,run

Slice it: group=='single'|'pair'|'triple', or k in {1,2,3}; task in
{classification, regression, static_link_prediction}. Stdlib only.

Usage:
    python scripts/experiments/analysis/multitask_ssl_pairs/build_combined_csv.py \
        --plotting-root results --out results/combined_all_arms.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# arm -> objectives it rotates over (defines group + k)
CONTAINS = {
    "NM": {"nm"}, "CL": {"cl"}, "FP": {"fp"},
    "NMCL": {"nm", "cl"}, "NMFP": {"nm", "fp"}, "CLFP": {"cl", "fp"},
    "MIX": {"nm", "cl", "fp"},
}
GROUP_OF_K = {1: "single", 2: "pair", 3: "triple"}
MODEL_ORDER = ["NM", "CL", "FP", "NMCL", "NMFP", "CLFP", "MIX"]
TASK_ORDER = ["classification", "regression", "static_link_prediction"]

# per-source-file: (relative csv path, canonical task name)
SOURCES = [
    ("node_classification/data/node_classification.csv", "classification"),
    ("node_regression/data/node_regression.csv", "regression"),
    ("static_link_prediction/data/static_link_prediction.csv", "static_link_prediction"),
]

KEY_COLS = ["group", "k", "model", "task", "dataset", "target", "split", "shots"]
METRIC_COLS = ["roc_auc", "accuracy", "f1", "spearman", "rmse", "mae", "r2", "mse"]
OUT_COLS = KEY_COLS + METRIC_COLS + ["run"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plotting-root", default="results")
    ap.add_argument("--out", default="results/combined_all_arms.csv")
    args = ap.parse_args()
    root = Path(args.plotting_root)

    rows: list[dict] = []
    skipped_models: set[str] = set()
    for rel, task in SOURCES:
        path = root / rel
        if not path.exists():
            print(f"WARN: missing {path}")
            continue
        with path.open() as fh:
            for r in csv.DictReader(fh):
                model = r.get("model", "")
                if model not in CONTAINS:
                    skipped_models.add(model)
                    continue
                k = len(CONTAINS[model])
                out = {c: "" for c in OUT_COLS}
                out.update({
                    "group": GROUP_OF_K[k], "k": k, "model": model, "task": task,
                    "dataset": r.get("dataset", ""), "target": r.get("target", ""),
                    "split": r.get("split", ""), "shots": r.get("shots", ""),
                    "run": r.get("run", ""),
                })
                for m in METRIC_COLS:
                    if r.get(m, "") not in ("", None):
                        out[m] = r[m]
                rows.append(out)

    rows.sort(key=lambda x: (
        TASK_ORDER.index(x["task"]) if x["task"] in TASK_ORDER else 99,
        x["k"],
        MODEL_ORDER.index(x["model"]) if x["model"] in MODEL_ORDER else 99,
        x["dataset"], x["target"], x["split"],
    ))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(rows)

    n_arms = len({r["model"] for r in rows})
    n_tasks = len({r["task"] for r in rows})
    print(f"wrote {out_path}: {len(rows)} rows | {n_arms} arms | {n_tasks} tasks")
    if skipped_models:
        print(f"  (skipped non-lattice models: {sorted(skipped_models)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
