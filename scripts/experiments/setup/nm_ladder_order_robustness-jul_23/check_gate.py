#!/usr/bin/env python3
"""Equivalence gate: does the source-subset shortcut reproduce a known ladder rung?

Order A rung 4 was trained on a purpose-built 4-source merge and published in
analysis/transfer/ladders/prodigy_nm/baseline/nm_ladder/data/nm_ladder_full.csv. train_gate_ordA_r4.yaml retrains that same
rung on the all8 graph restricted to those 4 sources. If the two agree, the shortcut is
sound and the other 11 rungs are worth launching; if they do not, every rung built on it
is invalid.

Reads the gate's eval log dirs the same way assemble_full_table.py does:
  log/eval_nm_ladder_gate_ordA_r4_to_<test>_nm_3shot_30way_<ts>/data/metrics_test*.json

Usage:
    python3 check_gate.py --log-root /dataMeR1/phil/gfm/prodigy/log
    python3 check_gate.py --log-root ... --tol 0.015
"""
import argparse
import json
import re
import sys
from pathlib import Path

CANON = [
    "ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
    "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter",
]

# Published order-A rung 4 (within_balanced, matched-40k), from
# analysis/transfer/ladders/prodigy_nm/baseline/nm_ladder/data/nm_ladder_full.csv. Used if the CSV is not readable.
PUBLISHED_R4 = [0.9344, 0.9753, 0.9093, 0.9113, 0.8297, 0.7768, 0.9234, 0.7235]

GATE_PREFIX = "nm_ladder_gate_ordA_r4"
RUN_RE = re.compile(
    rf"^eval_{GATE_PREFIX}_to_(?P<test>.+?)_nm_(?P<shots>\d+)shot_(?P<nway>\d+)way"
)


def step_of(path):
    m = re.search(r"_step(\d+)\.json$", path.name)
    return int(m.group(1)) if m else -1


def latest_roc_auc(run_dir):
    data_dir = run_dir / "data"
    if not data_dir.is_dir():
        return None
    for path in sorted(data_dir.glob("metrics_test*.json"), key=step_of, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        val = payload.get("test_roc_auc")
        if val is not None:
            return float(val)
    return None


def load_gate(log_root, shots, nway):
    cells = {}
    if not log_root.is_dir():
        return cells
    for run_dir in sorted(log_root.glob(f"eval_{GATE_PREFIX}_to_*_nm_{shots}shot_{nway}way*")):
        if not run_dir.is_dir():
            continue
        m = RUN_RE.match(run_dir.name)
        if not m or m["test"] not in CANON:
            continue
        val = latest_roc_auc(run_dir)
        if val is not None:
            cells[m["test"]] = val  # newest dir wins (glob sorted ascending)
    return cells


def load_published(csv_path):
    """Rung-4 within_balanced row from the ladder CSV, or the baked-in fallback."""
    try:
        import csv as _csv
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                if row.get("rung") == "4" and row.get("sampling") == "within_balanced":
                    return [float(row[c]) for c in CANON], str(csv_path)
    except (OSError, KeyError, ValueError):
        pass
    return PUBLISHED_R4, "baked-in fallback"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-root", type=Path, default=Path("/dataMeR1/phil/gfm/prodigy/log"))
    # parents[2] == scripts/experiments (this file is setup/<exp>/check_gate.py)
    ap.add_argument("--csv", type=Path,
                    default=Path(__file__).resolve().parents[2]
                    / "analysis/transfer/ladders/prodigy_nm/baseline/nm_ladder/data/nm_ladder_full.csv")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="max acceptable |delta| per column (default 0.01)")
    ap.add_argument("--shots", default="3")
    ap.add_argument("--nway", default="30")
    args = ap.parse_args()

    published, source = load_published(args.csv)
    gate = load_gate(args.log_root, args.shots, args.nway)

    print(f"reference: order A rung 4 ({source})")
    print(f"gate logs: {args.log_root}\n")

    missing = [c for c in CANON if c not in gate]
    if missing:
        print(f"MISSING gate evals for: {', '.join(missing)}")
        if len(missing) == len(CANON):
            print("\nNo gate eval logs found at all -- has the gate been trained and evaluated?")
            return 2

    print(f"{'test graph':<20} {'published':>10} {'subset':>10} {'delta':>9}")
    deltas = []
    for col in CANON:
        if col not in gate:
            print(f"{col:<20} {published[CANON.index(col)]:>10.4f} {'--':>10} {'--':>9}")
            continue
        pub = published[CANON.index(col)]
        got = gate[col]
        d = got - pub
        deltas.append(d)
        flag = "" if abs(d) <= args.tol else "   <-- over tol"
        print(f"{col:<20} {pub:>10.4f} {got:>10.4f} {d:>+9.4f}{flag}")

    if not deltas:
        return 2

    # Two independent ways the shortcut can be wrong, so two criteria:
    #   scatter    -- any single column off by more than tol
    #   systematic -- every column shifted the same way. A uniform offset is not seed
    #                 noise; it is what a genuine non-equivalence looks like (a
    #                 different sampling stream, a changed episode budget), and it can
    #                 hide under a per-column tolerance. Held to half the tolerance.
    max_abs = max(abs(d) for d in deltas)
    mean_signed = sum(deltas) / len(deltas)
    bias_tol = args.tol / 2
    scatter_ok = max_abs <= args.tol
    systematic_ok = abs(mean_signed) <= bias_tol

    print(f"\nmax |delta|        = {max_abs:.4f}   (tolerance {args.tol})"
          f"   {'ok' if scatter_ok else 'FAIL'}")
    print(f"mean signed delta  = {mean_signed:+.4f}   (tolerance +/-{bias_tol})"
          f"   {'ok' if systematic_ok else 'FAIL'}")

    if missing:
        print("\nINCOMPLETE: some columns were not evaluated; not a pass.")
        return 2
    if scatter_ok and systematic_ok:
        print("\nPASS -- the subset shortcut reproduces the published rung. Launch the 11 rungs.")
        return 0
    if not systematic_ok:
        print(f"\n  {sum(1 for d in deltas if d > 0)}/{len(deltas)} columns shifted positive; "
              "a one-sided shift is a systematic offset, not seed noise.")
    print("\nFAIL -- the shortcut does not reproduce the published rung. Do NOT launch the\n"
          "        11 rungs; fall back to building per-order nested merges.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
