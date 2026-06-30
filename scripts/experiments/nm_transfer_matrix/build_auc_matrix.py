#!/usr/bin/env python3
"""Aggregate NM transfer-matrix eval logs into a train x test AUC matrix.

Reads the eval log dirs produced by eval_nm_matrix_tucker.sh
(``eval_<train>_to_<test>_nm_0shot_<timestamp>/data/metrics_test*.json``),
extracts ``test_roc_auc``, and pivots into a train(row) x test(col) matrix.

Usage:
    python build_auc_matrix.py --log-root /dataMeR1/phil/gfm/prodigy/log \
        --out-csv auc_matrix.csv

Stdlib only (no pandas/numpy needed).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

RUN_RE = re.compile(
    r"^eval_(?P<model>.+?)_to_(?P<dataset>.+?)_nm_(?P<shots>\d+)shot(?:_(?P<ts>\d.*))?$"
)

# Display labels (rows = train source, cols = test source).
MODEL_LABELS = {
    "nm_matrix_ukr": "ukr",
    "nm_matrix_covid": "covid",
    "nm_matrix_merged_match": "merged @match",
    "nm_matrix_merged_full": "merged @full",
    "nm_matrix_merged": "merged",  # fallback (single-checkpoint runs)
}
DATASET_LABELS = {
    "ukr_rus_twitter": "ukr",
    "covid19_twitter": "covid",
    "merged_ukr_rus_covid": "merged",
}
ROW_ORDER = ["ukr", "covid", "merged @match", "merged @full", "merged"]
COL_ORDER = ["ukr", "covid", "merged"]


def step_of(path: Path) -> int:
    m = re.search(r"_step(\d+)\.json$", path.name)
    return int(m.group(1)) if m else -1


def latest_metric(run_dir: Path, metric: str) -> float | None:
    """Return test_<metric> from the highest-step metrics_test*.json, if any."""
    key = f"test_{metric}"
    data_dir = run_dir / "data"
    candidates = sorted(data_dir.glob("metrics_test*.json"), key=step_of)
    for path in reversed(candidates):  # prefer highest step / plain `metrics_test.json`
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        val = payload.get(key)
        if val is not None:
            return float(val)
    return None


def collect(log_root: Path, shots: str = "3", nway: str = "30",
            metric: str = "roc_auc") -> dict[tuple[str, str], float]:
    cells: dict[tuple[str, str], float] = {}
    for run_dir in sorted(log_root.glob(f"eval_nm_matrix_*_to_*_nm_{shots}shot_{nway}way*")):
        if not run_dir.is_dir():
            continue
        m = RUN_RE.match(run_dir.name)
        if not m:
            continue
        row = MODEL_LABELS.get(m["model"], m["model"])
        col = DATASET_LABELS.get(m["dataset"], m["dataset"])
        val = latest_metric(run_dir, metric)
        if val is None:
            print(f"[warn] no test_{metric} in {run_dir.name}")
            continue
        # Multiple timestamps for the same cell -> keep the newest run dir
        # (glob is sorted, so the later one wins by overwriting).
        cells[(row, col)] = val
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--log-root",
        default="log",
        help="Directory holding eval_* run dirs (default: ./log).",
    )
    ap.add_argument("--out-csv", default=None, help="Optional path to write the matrix CSV (long format with a metric column).")
    ap.add_argument("--shots", default="3", help="Which n_shot eval to read (NM is degenerate at 0-shot).")
    ap.add_argument("--n-way", default="30", help="Which n_way eval to read (3-way is near-ceiling; 30 is discriminative).")
    ap.add_argument("--metric", default="all", choices=["roc_auc", "accuracy", "f1", "all"],
                    help="Which test metric(s) to report (default: all three).")
    args = ap.parse_args()

    log_root = Path(args.log_root)
    if not log_root.is_dir():
        raise SystemExit(f"log-root not found: {log_root}")

    metrics = ["roc_auc", "accuracy", "f1"] if args.metric == "all" else [args.metric]
    csv_rows: list[list[str]] = []

    for metric in metrics:
        cells = collect(log_root, args.shots, args.n_way, metric)
        if not cells:
            print(f"[warn] no eval dirs with test_{metric} found")
            continue

        rows = [r for r in ROW_ORDER if any(k[0] == r for k in cells)] or sorted({k[0] for k in cells})
        cols = [c for c in COL_ORDER if any(k[1] == c for k in cells)] or sorted({k[1] for k in cells})

        def cell(r, c):
            return cells.get((r, c))

        print(f"\n=== {metric} (train rows / test cols) ===")
        header = "train\\test".ljust(12) + "".join(c.ljust(10) for c in cols)
        print(header)
        print("-" * len(header))
        for r in rows:
            line = r.ljust(12)
            for c in cols:
                v = cell(r, c)
                line += (f"{v:.4f}" if v is not None else "  -   ").ljust(10)
            print(line)
            for c in cols:
                if cell(r, c) is not None:
                    csv_rows.append([metric, r, c, f"{cell(r, c):.6f}"])

        # Highlight the inversion of interest (single vs merged, cross-domain).
        # Check whichever merged rows are present (@match / @full / plain).
        merged_rows = [m for m in ("merged @match", "merged @full", "merged") if any(k[0] == m for k in cells)]
        for tgt, single in (("covid", "ukr"), ("ukr", "covid")):
            s = cell(single, tgt)
            for mrow in merged_rows:
                mg = cell(mrow, tgt)
                if s is not None and mg is not None:
                    verdict = "INVERSION reproduced" if s > mg else "no inversion"
                    print(f"  test={tgt}: single({single})={s:.4f} vs {mrow}={mg:.4f} "
                          f"(Δ={s - mg:+.4f}) -> {verdict}")

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
