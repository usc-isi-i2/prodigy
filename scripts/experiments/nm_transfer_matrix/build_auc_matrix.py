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
    "nm_matrix_merged": "merged",
}
DATASET_LABELS = {
    "ukr_rus_twitter": "ukr",
    "covid19_twitter": "covid",
    "merged_ukr_rus_covid": "merged",
}
ROW_ORDER = ["ukr", "covid", "merged"]
COL_ORDER = ["ukr", "covid", "merged"]


def step_of(path: Path) -> int:
    m = re.search(r"_step(\d+)\.json$", path.name)
    return int(m.group(1)) if m else -1


def latest_test_auc(run_dir: Path) -> float | None:
    """Return test_roc_auc from the highest-step metrics_test*.json, if any."""
    data_dir = run_dir / "data"
    candidates = sorted(data_dir.glob("metrics_test*.json"), key=step_of)
    for path in reversed(candidates):  # prefer highest step / plain `metrics_test.json`
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        auc = payload.get("test_roc_auc")
        if auc is not None:
            return float(auc)
    return None


def collect(log_root: Path, shots: str = "3") -> dict[tuple[str, str], float]:
    cells: dict[tuple[str, str], float] = {}
    for run_dir in sorted(log_root.glob(f"eval_nm_matrix_*_to_*_nm_{shots}shot*")):
        if not run_dir.is_dir():
            continue
        m = RUN_RE.match(run_dir.name)
        if not m:
            continue
        row = MODEL_LABELS.get(m["model"], m["model"])
        col = DATASET_LABELS.get(m["dataset"], m["dataset"])
        auc = latest_test_auc(run_dir)
        if auc is None:
            print(f"[warn] no test_roc_auc in {run_dir.name}")
            continue
        prev = cells.get((row, col))
        # Multiple timestamps for the same cell -> keep the newest run dir
        # (glob is sorted, so the later one wins by overwriting).
        cells[(row, col)] = auc
        if prev is not None:
            print(f"[info] {row}->{col}: replaced {prev:.4f} with newer {auc:.4f}")
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--log-root",
        default="log",
        help="Directory holding eval_* run dirs (default: ./log).",
    )
    ap.add_argument("--out-csv", default=None, help="Optional path to write the matrix CSV.")
    ap.add_argument("--shots", default="3", help="Which n_shot eval to read (NM is degenerate at 0-shot).")
    args = ap.parse_args()

    log_root = Path(args.log_root)
    if not log_root.is_dir():
        raise SystemExit(f"log-root not found: {log_root}")

    cells = collect(log_root, args.shots)
    if not cells:
        raise SystemExit(f"No NM transfer-matrix eval dirs found under {log_root}")

    rows = [r for r in ROW_ORDER if any(k[0] == r for k in cells)] or sorted({k[0] for k in cells})
    cols = [c for c in COL_ORDER if any(k[1] == c for k in cells)] or sorted({k[1] for k in cells})

    # Pretty print
    header = "train\\test".ljust(12) + "".join(c.ljust(10) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = r.ljust(12)
        for c in cols:
            v = cells.get((r, c))
            line += (f"{v:.4f}" if v is not None else "  -   ").ljust(10)
        print(line)

    # Highlight the inversion of interest
    def cell(r, c):
        return cells.get((r, c))

    print()
    for tgt, single in (("covid", "ukr"), ("ukr", "covid")):
        s, mg = cell(single, tgt), cell("merged", tgt)
        if s is not None and mg is not None:
            verdict = "INVERSION reproduced" if s > mg else "no inversion"
            print(
                f"test={tgt}: single({single})={s:.4f} vs merged={mg:.4f} "
                f"(Δ={s - mg:+.4f}) -> {verdict}"
            )

    if args.out_csv:
        out = Path(args.out_csv)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["train\\test", *cols])
            for r in rows:
                w.writerow([r, *("" if cell(r, c) is None else f"{cell(r, c):.6f}" for c in cols)])
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
