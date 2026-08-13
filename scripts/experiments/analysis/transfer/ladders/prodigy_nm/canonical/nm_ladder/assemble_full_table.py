#!/usr/bin/env python3
"""Assemble the COMPLETE 8-rung NM interpolation ladder.

Pulls the existing rungs (1 ukr, 2 ukr+cov, 3 ukr+cov+mid, 8 all8) from the
published ladder CSV and the 4 newly-trained fill-in rungs (4..7) from their eval
log dirs, then emits one 8x8 train(rung) x test(graph) AUC table.

New-rung eval dirs are produced by eval_ladder_tucker.sh and look like
``eval_nm_ladder_<N>src_to_<test>_nm_3shot_30way_<ts>/data/metrics_test*.json``
(``test_roc_auc`` is read from the highest-step file).

Outputs (into --out-dir, default this script's dir):
  - ``nm_ladder_full.csv`` — rung,n_sources,train_graph,added,sampling,<8 dataset cols>
  - a pretty-printed 8x8 with the "staircase" jump for each newly-added column.

Stdlib only.

Usage (on Tucker, where both the logs and the repo CSV live):
    python assemble_full_table.py --log-root /dataMeR1/phil/gfm/prodigy/log
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# Canonical column order (matches nmladder_results.csv / build_ss_matrix.py).
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

# Rung layout: (rung, n_sources, train_label, column_that_enters_at_this_rung).
# The `added` column is the graph that FIRST enters training at that rung — its
# test column is expected to jump vs the rung below (the interpolation signal).
RUNGS = [
    (1, 1, "ukr", "ukr_rus_twitter"),
    (2, 2, "ukr+cov", "covid19_twitter"),
    (3, 3, "ukr+cov+mid", "midterm"),
    (4, 4, "+cov_pol", "covid_political"),
    (5, 5, "+elec20", "election2020"),
    (6, 6, "+ukr_susp", "ukr_rus_suspended"),
    (7, 7, "+twibot20", "twibot20"),
    (8, 8, "all8", "cp_hk_twitter"),
]
NEW_RUNGS = {4, 5, 6, 7}  # trained here; the rest come from the ladder CSV

# Fallback for the existing rungs if the ladder CSV is absent (e.g. it's gitignored
# and wasn't synced to Tucker). These are the published within-balanced values from
# scripts/experiments/analysis/transfer/ladders/prodigy_nm/canonical/nm_ladder/data/nmladder_results.csv (matched-40k, NM 3-shot 30-way).
# The CSV, when present, always wins over this.
FALLBACK_EXISTING = {
    1: [0.948, 0.973, 0.874, 0.849, 0.828, 0.771, 0.921, 0.724],
    2: [0.945, 0.980, 0.885, 0.843, 0.828, 0.775, 0.925, 0.726],
    3: [0.941, 0.978, 0.915, 0.830, 0.815, 0.777, 0.927, 0.720],
    8: [0.934, 0.975, 0.908, 0.906, 0.920, 0.931, 0.937, 0.867],
}
# ladder-CSV train_graph label -> full-ladder rung number (within_balanced rows only).
CSV_LABEL_TO_RUNG = {"ukr": 1, "ukr+cov": 2, "ukr+cov+mid": 3, "all8": 8}

RUN_RE = re.compile(
    r"^eval_nm_ladder_(?P<n>\d+)src_to_(?P<test>.+?)_nm_(?P<shots>\d+)shot_(?P<nway>\d+)way"
)


def step_of(path: Path) -> int:
    m = re.search(r"_step(\d+)\.json$", path.name)
    return int(m.group(1)) if m else -1


def latest_roc_auc(run_dir: Path) -> float | None:
    data_dir = run_dir / "data"
    for path in sorted(data_dir.glob("metrics_test*.json"), key=step_of, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        val = payload.get("test_roc_auc")
        if val is not None:
            return float(val)
    return None


def load_existing_from_csv(csv_path: Path) -> dict[int, list[float]]:
    """rung -> 8 AUCs, from the within_balanced rows of the ladder CSV."""
    out: dict[int, list[float]] = {}
    if not csv_path.is_file():
        return out
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("sampling") != "within_balanced":
                continue
            rung = CSV_LABEL_TO_RUNG.get(row.get("train_graph", ""))
            if rung is None:
                continue
            try:
                out[rung] = [float(row[c]) for c in CANON]
            except (KeyError, ValueError):
                print(f"[warn] malformed CSV row for {row.get('train_graph')!r}")
    return out


def load_single_source_csv(ss_csv: Path) -> list[tuple[str, list[float]]]:
    """Single-source specialist rows (train ONE graph, test all 8), from the
    nm_single_source_matrix wide CSV. Returns [(train_key, 8 aucs)] in CANON order."""
    rows: dict[str, list[float]] = {}
    if not ss_csv.is_file():
        return []
    with ss_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            train = row.get("train_graph")
            if train not in CANON:
                continue
            try:
                rows[train] = [float(row[c]) for c in CANON]
            except (KeyError, ValueError):
                print(f"[warn] malformed single-source CSV row for {train!r}")
    return [(t, rows[t]) for t in CANON if t in rows]


def load_new_from_logs(log_root: Path, shots: str, nway: str) -> dict[int, dict[str, float]]:
    """rung -> {test_col: auc} for the fill-in rungs, from eval log dirs."""
    cells: dict[int, dict[str, float]] = {}
    if not log_root.is_dir():
        return cells
    for run_dir in sorted(log_root.glob(f"eval_nm_ladder_*src_to_*_nm_{shots}shot_{nway}way*")):
        if not run_dir.is_dir():
            continue
        m = RUN_RE.match(run_dir.name)
        if not m:
            continue
        rung = int(m["n"])
        test = m["test"]
        if test not in CANON:
            print(f"[warn] unknown test dataset {test!r} in {run_dir.name}")
            continue
        val = latest_roc_auc(run_dir)
        if val is None:
            print(f"[warn] no test_roc_auc in {run_dir.name}")
            continue
        cells.setdefault(rung, {})[test] = val  # newest dir wins (glob sorted ascending)
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    repo_root = next(p for p in here.parents if (p / "AGENTS.md").is_file())
    ap.add_argument("--log-root", default="log",
                    help="Dir holding eval_nm_ladder_*src_* run dirs (Tucker: "
                         "/dataMeR1/phil/gfm/prodigy/log).")
    ap.add_argument("--ladder-csv",
                    default=str(repo_root / "scripts/experiments/analysis/transfer/ladders/prodigy_nm/canonical/nm_ladder/data/nmladder_results.csv"),
                    help="Published ladder CSV for the existing rungs 1/2/3/8.")
    ap.add_argument("--ss-csv",
                    default=str(repo_root / "scripts/experiments/analysis/transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix/data/nm_single_source_matrix.csv"),
                    help="Single-source matrix CSV; if present, its 8 specialist rows are "
                         "appended to the combined table + a combined CSV is written.")
    ap.add_argument("--shots", default="3")
    ap.add_argument("--n-way", default="30")
    ap.add_argument("--out-dir", default=str(here))
    args = ap.parse_args()

    existing = load_existing_from_csv(Path(args.ladder_csv))
    if existing:
        print(f"[ok] existing rungs from {args.ladder_csv}: {sorted(existing)}")
    else:
        print(f"[warn] ladder CSV not found/empty ({args.ladder_csv}); using embedded "
              f"fallback for rungs 1/2/3/8")
        existing = {k: list(v) for k, v in FALLBACK_EXISTING.items()}

    new_cells = load_new_from_logs(Path(args.log_root), args.shots, args.n_way)
    print(f"[ok] fill-in rungs found in logs: {sorted(new_cells) or 'NONE'}")

    # Assemble rung -> 8 values (None where missing).
    table: dict[int, list[float | None]] = {}
    for rung, n_src, label, added in RUNGS:
        if rung in NEW_RUNGS:
            row = new_cells.get(rung, {})
            table[rung] = [row.get(c) for c in CANON]
        else:
            vals = existing.get(rung)
            table[rung] = list(vals) if vals else [None] * len(CANON)

    # Pretty print.
    print("\n=== NM interpolation ladder (roc_auc, train rungs / test cols) ===")
    header = "rung  train".ljust(20) + "".join(SHORT[c].ljust(9) for c in CANON)
    print(header)
    print("-" * len(header))
    for rung, n_src, label, added in RUNGS:
        line = f"{rung} ({n_src})  {label}".ljust(20)
        for col, c in enumerate(CANON):
            v = table[rung][col]
            mark = "+" if c == added else " "  # + marks the column that enters here
            line += (f"{v:.3f}{mark}" if v is not None else "  -  ").ljust(9)
        print(line)
    print("  (+ = the graph that first enters training at that rung)")

    # Staircase diagnostic: newly-added column, this rung vs the rung below.
    print("\n=== staircase: does the added column jump when its graph enters? ===")
    for rung, n_src, label, added in RUNGS:
        if rung == 1:
            continue
        col = CANON.index(added)
        cur = table[rung][col]
        prev = table[rung - 1][col]
        if cur is None or prev is None:
            print(f"  rung {rung} {SHORT[added]:9s}: (incomplete)")
            continue
        print(f"  rung {rung} {SHORT[added]:9s}: {prev:.3f} -> {cur:.3f}  "
              f"(Δ {cur - prev:+.3f}) on entering")

    # Write the full CSV.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nm_ladder_full.csv"
    n_missing = 0
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rung", "n_sources", "train_graph", "added", "sampling"] + CANON)
        for rung, n_src, label, added in RUNGS:
            cells = []
            for v in table[rung]:
                if v is None:
                    n_missing += 1
                cells.append(f"{v:.4f}" if v is not None else "")
            w.writerow([rung, n_src, label, added, "within_balanced"] + cells)
    print(f"\nwrote {out_path}" + (f"  ({n_missing} empty cells — some evals missing)" if n_missing else ""))

    # Optionally extend with the single-source specialist rows (train ONE graph, test all 8).
    ss = load_single_source_csv(Path(args.ss_csv))
    if ss:
        print(f"\n[ok] single-source specialists from {args.ss_csv}: {len(ss)} rows")
        print("\n=== + single-source specialists (roc_auc; * = in-domain, train==test) ===")
        header = "train".ljust(11) + "".join(SHORT[c].ljust(9) for c in CANON)
        print(header)
        print("-" * len(header))
        for train, vals in ss:
            line = SHORT[train].ljust(11)
            for col, c in enumerate(CANON):
                mark = "*" if c == train else " "
                line += (f"{vals[col]:.3f}{mark}").ljust(9)
            print(line)

        combined_path = out_dir / "nm_ladder_plus_single_source.csv"
        with combined_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["block", "row", "diag_or_added"] + CANON)
            for rung, n_src, label, added in RUNGS:
                cells = [f"{v:.4f}" if v is not None else "" for v in table[rung]]
                w.writerow(["ladder", f"{rung}:{label}", added] + cells)
            for train, vals in ss:
                w.writerow(["single_source", SHORT[train], train] + [f"{v:.4f}" for v in vals])
        print(f"\nwrote {combined_path}")
    else:
        print(f"\n[note] no single-source CSV at {args.ss_csv}; skipping the specialist block")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
