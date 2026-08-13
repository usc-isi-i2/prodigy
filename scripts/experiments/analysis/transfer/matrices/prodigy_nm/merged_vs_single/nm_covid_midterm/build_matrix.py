#!/usr/bin/env python3
"""Aggregate the cov/mid eval logs into train x test tables (acc/f1/auc).

Reads eval dirs `eval_<model>_to_<dataset>_nm_<shots>shot_<nway>way_*` for the 5
cov/mid regimes and 3 test sets, and prints one table per metric.

    python build_matrix.py --log-root /dataMeR1/phil/gfm/prodigy/log \
        --shots 3 --n-way 30 --metric all --out-csv matrix.csv

Stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

RUN_RE = re.compile(r"^eval_(?P<model>.+?)_to_(?P<dataset>.+?)_nm_(?P<shots>\d+)shot_(?P<nway>\d+)way(?:_.*)?$")

# Merged models are evaluated at two checkpoints: @match (same step count as the
# single-source runs = matched total compute) and @full (final = matched per-domain
# exposure). Single-source models exist only at their final step.
MODEL_LABELS = {
    "nm_cm_midterm": "midterm",
    "nm_cm_covid": "covid",
    "nm_cm_merged_match": "merged-naive @match",
    "nm_cm_merged_full": "merged-naive @full",
    "nm_cm_within_match": "merged-within @match",
    "nm_cm_within_full": "merged-within @full",
    "nm_cm_within_balanced_match": "merged-within-bal @match",
    "nm_cm_within_balanced_full": "merged-within-bal @full",
    # fall back to single-checkpoint names if make_model_list wasn't used
    "nm_cm_merged": "merged-naive",
    "nm_cm_within": "merged-within",
    "nm_cm_within_balanced": "merged-within-balanced",
}
DATASET_LABELS = {
    "midterm": "midterm",
    "covid19_twitter": "covid",
    "ukr_rus_twitter": "ukr (held-out)",
    "merged_covid_midterm": "merged",
}
ROW_ORDER = [
    "midterm", "covid",
    "merged-naive @match", "merged-naive @full",
    "merged-within @match", "merged-within @full",
    "merged-within-bal @match", "merged-within-bal @full",
    "merged-naive", "merged-within", "merged-within-balanced",
]
COL_ORDER = ["midterm", "covid", "ukr (held-out)", "merged"]


def step_of(p: Path) -> int:
    m = re.search(r"_step(\d+)\.json$", p.name)
    return int(m.group(1)) if m else -1


def latest_metric(run_dir: Path, metric: str):
    key = f"test_{metric}"
    for p in sorted((run_dir / "data").glob("metrics_test*.json"), key=step_of, reverse=True):
        try:
            v = json.loads(p.read_text()).get(key)
        except (OSError, json.JSONDecodeError):
            continue
        if v is not None:
            return float(v)
    return None


def collect(log_root: Path, shots: str, nway: str, metric: str):
    cells = {}
    for run_dir in sorted(log_root.glob(f"eval_nm_cm_*_to_*_nm_{shots}shot_{nway}way*")):
        if not run_dir.is_dir():
            continue
        m = RUN_RE.match(run_dir.name)
        if not m:
            continue
        row = MODEL_LABELS.get(m["model"], m["model"])
        col = DATASET_LABELS.get(m["dataset"], m["dataset"])
        v = latest_metric(run_dir, metric)
        if v is not None:
            cells[(row, col)] = v
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-root", default="log")
    ap.add_argument("--shots", default="3")
    ap.add_argument("--n-way", default="30")
    ap.add_argument("--metric", default="all", choices=["roc_auc", "accuracy", "f1", "all"])
    ap.add_argument("--out-csv", default=None, help="Optional long-format CSV (metric,train,test,value).")
    args = ap.parse_args()

    log_root = Path(args.log_root)
    if not log_root.is_dir():
        raise SystemExit(f"log-root not found: {log_root}")

    metrics = ["roc_auc", "accuracy", "f1"] if args.metric == "all" else [args.metric]
    csv_rows: list[list[str]] = []
    any_found = False

    for metric in metrics:
        cells = collect(log_root, args.shots, args.n_way, metric)
        if not cells:
            print(f"[warn] no eval dirs with test_{metric}")
            continue
        any_found = True
        rows = [r for r in ROW_ORDER if any(k[0] == r for k in cells)] or sorted({k[0] for k in cells})
        cols = [c for c in COL_ORDER if any(k[1] == c for k in cells)] or sorted({k[1] for k in cells})

        print(f"\n=== {metric} (train rows / test cols) ===")
        header = "train\\test".ljust(24) + "".join(c.ljust(10) for c in cols)
        print(header)
        print("-" * len(header))
        for r in rows:
            line = r.ljust(24)
            for c in cols:
                v = cells.get((r, c))
                line += (f"{v:.4f}" if v is not None else "  -   ").ljust(10)
                if v is not None:
                    csv_rows.append([metric, r, c, f"{v:.6f}"])
            print(line)

    if not any_found:
        raise SystemExit(f"No cov/mid eval dirs found under {log_root}")

    if args.out_csv and csv_rows:
        out = Path(args.out_csv)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "train", "test", "value"])
            w.writerows(csv_rows)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
