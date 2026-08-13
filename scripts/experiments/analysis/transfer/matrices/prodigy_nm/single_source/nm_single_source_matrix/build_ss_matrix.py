#!/usr/bin/env python3
"""Aggregate the single-source NM eval logs into an 8x8 train x test matrix.

Reads the eval run dirs produced by eval_matrix_tucker.sh
(``eval_nm_ss_<train>_to_<test>_nm_3shot_30way_<ts>/data/metrics_test*.json``),
extracts ``test_roc_auc`` (and accuracy / f1), and pivots into a train(row) x
test(col) matrix over the 8 single-source graphs.

Outputs:
  - a pretty-printed matrix per metric (diagonal = in-domain specialist);
  - ``nm_single_source_matrix.csv`` — WIDE, roc_auc only, columns = the 8 canonical
    dataset keys, one row per train source. Same column shape as
    ``scripts/experiments/analysis/transfer/ladders/prodigy_nm/canonical/nm_ladder/data/nmladder_results.csv`` so the single-source rows
    can be concatenated with the merged-ladder rows for one combined figure.
  - ``nm_single_source_matrix_long.csv`` — LONG, all three metrics.

Stdlib only.

Usage:
    python build_ss_matrix.py --log-root /dataMeR1/phil/gfm/prodigy/log
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# eval_nm_ss_<train>_to_<test>_nm_<shots>shot_<nway>way[_<timestamp>]
RUN_RE = re.compile(
    r"^eval_nm_ss_(?P<train>.+?)_to_(?P<test>.+?)_nm_(?P<shots>\d+)shot_(?P<nway>\d+)way"
)

# Canonical column order (matches the NM ladder table / nmladder_results.csv).
CANON = [
    "ukr_rus_twitter",
    "covid19_twitter",
    "midterm",
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
    "cp_hk_twitter",
]
SHORT = {
    "ukr_rus_twitter": "ukr",
    "covid19_twitter": "covid",
    "midterm": "midterm",
    "covid_political": "cov_pol",
    "election2020": "elec20",
    "ukr_rus_suspended": "ukr_susp",
    "twibot20": "twibot20",
    "cp_hk_twitter": "cp_hk",
}
METRICS = ["roc_auc", "accuracy", "f1"]


def step_of(path: Path) -> int:
    m = re.search(r"_step(\d+)\.json$", path.name)
    return int(m.group(1)) if m else -1


def latest_metric(run_dir: Path, metric: str) -> float | None:
    """Return test_<metric> from the highest-step metrics_test*.json, if any."""
    key = f"test_{metric}"
    data_dir = run_dir / "data"
    for path in sorted(data_dir.glob("metrics_test*.json"), key=step_of, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        val = payload.get(key)
        if val is not None:
            return float(val)
    return None


def collect(log_root: Path, shots: str, nway: str, metric: str) -> dict[tuple[str, str], float]:
    cells: dict[tuple[str, str], float] = {}
    for run_dir in sorted(log_root.glob(f"eval_nm_ss_*_to_*_nm_{shots}shot_{nway}way*")):
        if not run_dir.is_dir():
            continue
        m = RUN_RE.match(run_dir.name)
        if not m:
            continue
        train, test = m["train"], m["test"]
        if test not in CANON:
            print(f"[warn] unknown test dataset {test!r} in {run_dir.name}")
            continue
        val = latest_metric(run_dir, metric)
        if val is None:
            print(f"[warn] no test_{metric} in {run_dir.name}")
            continue
        cells[(train, test)] = val  # newest run dir wins (glob sorted ascending)
    return cells


def print_matrix(cells: dict[tuple[str, str], float], metric: str) -> None:
    rows = [r for r in CANON if any(k[0] == r for k in cells)]
    print(f"\n=== {metric} (train rows / test cols) ===")
    header = "train\\test".ljust(11) + "".join(SHORT[c].ljust(9) for c in CANON)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = SHORT[r].ljust(11)
        for c in CANON:
            v = cells.get((r, c))
            marker = "*" if r == c and v is not None else " "  # * = in-domain diagonal
            line += ((f"{v:.3f}{marker}" if v is not None else "  -  ")).ljust(9)
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-root", default="log", help="Dir holding eval_nm_ss_* run dirs.")
    ap.add_argument("--shots", default="3")
    ap.add_argument("--n-way", default="30")
    ap.add_argument("--out-dir", default=None, help="Where to write CSVs (default: this script's dir).")
    args = ap.parse_args()

    log_root = Path(args.log_root)
    if not log_root.is_dir():
        raise SystemExit(f"log-root not found: {log_root}")
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent

    by_metric = {m: collect(log_root, args.shots, args.n_way, m) for m in METRICS}
    if not by_metric["roc_auc"]:
        raise SystemExit("no eval dirs with test_roc_auc found — check --log-root / that eval ran")

    for metric in METRICS:
        if by_metric[metric]:
            print_matrix(by_metric[metric], metric)

    # WIDE roc_auc CSV (same column shape as nmladder_results.csv).
    auc = by_metric["roc_auc"]
    wide_path = out_dir / "nm_single_source_matrix.csv"
    with wide_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["train_graph"] + CANON)
        for r in CANON:
            if not any(k[0] == r for k in auc):
                continue
            w.writerow([r] + [f"{auc[(r, c)]:.4f}" if (r, c) in auc else "" for c in CANON])
    print(f"\nwrote {wide_path}")

    # LONG CSV with all metrics.
    long_path = out_dir / "nm_single_source_matrix_long.csv"
    with long_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "train", "test", "value"])
        for metric in METRICS:
            for (r, c), v in sorted(by_metric[metric].items()):
                w.writerow([metric, r, c, f"{v:.6f}"])
    print(f"wrote {long_path}")

    # Cross-check against the ladder: ukr specialist should ~match ladder rung 1.
    diag = auc.get(("ukr_rus_twitter", "ukr_rus_twitter"))
    if diag is not None:
        print(f"\n[cross-check] ukr->ukr diagonal = {diag:.3f} "
              f"(ladder rung 1 ukr->ukr was .948; large gap => protocol drift)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
