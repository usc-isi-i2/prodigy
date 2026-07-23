#!/usr/bin/env python3
"""Assemble the order-robustness results into the deliverable CSVs.

Emits two files:

  nm_ladder_order_robustness.csv        wide: one row per (order, rung), 8 test columns
  nm_ladder_order_robustness_long.csv   entry-aligned long form, one row per cell

The long form is what the event-study figure needs: each cell carries `rel_to_entry`
(rung minus the rung at which that test graph joined the training merge), so curves from
different orders can be overlaid on a common x-axis. rel_to_entry < 0 is out-of-merge
(zero-shot transfer), >= 0 is in-merge.

Reused rungs are pulled from the published tables rather than retrained -- rung 8 is
order-invariant, rung 1 of any order is that graph's single-source specialist, and order
B rung 2 has the same source SET as order A rung 2.

Usage:
    python3 assemble_order_table.py --log-root /dataMeR1/phil/gfm/prodigy/log
    python3 assemble_order_table.py --log-root ... --allow-partial
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_configs import ORDERS, SOURCES, canonical, plan  # noqa: E402

CANON = [s[1] for s in SOURCES]          # dataset_key, in graph_id order == table column order
KEY_OF_DATASET = {s[1]: s[0] for s in SOURCES}   # dataset_key -> merge key
DATASET_OF_KEY = {s[0]: s[1] for s in SOURCES}


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


def eval_row(log_root, prefix, shots="3", nway="30"):
    """The 8 test-graph AUCs for one trained run, from its eval log dirs."""
    cells = {}
    pattern = re.compile(rf"^eval_{re.escape(prefix)}_to_(?P<test>.+?)_nm_{shots}shot_{nway}way")
    for run_dir in sorted(log_root.glob(f"eval_{prefix}_to_*_nm_{shots}shot_{nway}way*")):
        if not run_dir.is_dir():
            continue
        m = pattern.match(run_dir.name)
        if not m or m["test"] not in CANON:
            continue
        val = latest_roc_auc(run_dir)
        if val is not None:
            cells[m["test"]] = val      # newest dir wins (glob sorted ascending)
    return cells


def published_lookup(ladder_csv, ss_csv):
    """frozenset(source keys) -> (dict of 8 AUCs, provenance string)."""
    lookup = {}

    # Order-A ladder rungs: rung k covers the first k sources of order A.
    try:
        with ladder_csv.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("sampling") != "within_balanced":
                    continue
                try:
                    rung = int(row["rung"])
                except (KeyError, ValueError):
                    continue
                fs = frozenset(ORDERS["A"][:rung])
                lookup[fs] = ({c: float(row[c]) for c in CANON},
                              f"order-A ladder rung {rung} ({ladder_csv.name})")
    except (OSError, KeyError, ValueError) as exc:
        print(f"WARN: could not read {ladder_csv}: {exc}", file=sys.stderr)

    # Single-source specialists: one row per training graph.
    try:
        with ss_csv.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                train = row.get("train_graph")
                if train not in KEY_OF_DATASET:
                    continue
                fs = frozenset([KEY_OF_DATASET[train]])
                # Do not clobber a ladder entry (order-A rung 1 == the ukr specialist);
                # the ladder row is the one the published table used.
                lookup.setdefault(fs, ({c: float(row[c]) for c in CANON},
                                       f"single-source matrix ({ss_csv.name})"))
    except (OSError, KeyError, ValueError) as exc:
        print(f"WARN: could not read {ss_csv}: {exc}", file=sys.stderr)

    return lookup


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-root", type=Path, default=Path("/dataMeR1/phil/gfm/prodigy/log"))
    analysis = Path(__file__).resolve().parents[2] / "analysis"
    ap.add_argument("--ladder-csv", type=Path,
                    default=analysis / "nm_ladder/data/nm_ladder_full.csv")
    ap.add_argument("--ss-csv", type=Path,
                    default=analysis / "nm_single_source_matrix/data/nm_single_source_matrix.csv")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--allow-partial", action="store_true",
                    help="write what exists instead of exiting 1 on missing cells")
    args = ap.parse_args()

    lookup = published_lookup(args.ladder_csv, args.ss_csv)
    rows = plan()

    wide, long_rows, missing = [], [], []
    for r in rows:
        order, rung, sources = r["order"], r["rung"], r["sources"]
        if r["status"] == "new":
            cells = eval_row(args.log_root, r["run"])
            provenance = r["run"]
        else:
            hit = lookup.get(frozenset(sources))
            if hit is None:
                cells, provenance = {}, f"UNRESOLVED reuse ({r['run']})"
            else:
                cells, provenance = hit[0], hit[1]

        absent = [c for c in CANON if c not in cells]
        if absent:
            missing.append(f"{order} r{rung} ({provenance}): {', '.join(absent)}")

        wide.append(dict(
            order=order, rung=rung, n_sources=len(sources), added=r["added"],
            sources=" ".join(sources), status=r["status"], provenance=provenance,
            **{c: cells.get(c, "") for c in CANON},
        ))

        seq = ORDERS[order]
        for test in CANON:
            key = KEY_OF_DATASET[test]
            entry_rung = seq.index(key) + 1
            long_rows.append(dict(
                order=order, rung=rung, n_sources=len(sources), test_graph=test,
                test_canonical=canonical(key), auc=cells.get(test, ""),
                entry_rung=entry_rung, rel_to_entry=rung - entry_rung,
                in_training=int(rung >= entry_rung),
                added=r["added"], sources=" ".join(sources),
                status=r["status"], provenance=provenance,
            ))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wide_path = args.out_dir / "nm_ladder_order_robustness.csv"
    long_path = args.out_dir / "nm_ladder_order_robustness_long.csv"

    with wide_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["order", "rung", "n_sources", "added", "sources",
                                           "status", "provenance"] + CANON)
        w.writeheader()
        w.writerows(wide)

    with long_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["order", "rung", "n_sources", "test_graph",
                                           "test_canonical", "auc", "entry_rung",
                                           "rel_to_entry", "in_training", "added",
                                           "sources", "status", "provenance"])
        w.writeheader()
        w.writerows(long_rows)

    print(f"wrote {wide_path}  ({len(wide)} rows)")
    print(f"wrote {long_path}  ({len(long_rows)} cells)")

    # Staircase diagnostic: mean AUC just before vs just after each graph's entry.
    print("\nentry-aligned check (mean over the 8 test graphs, per order):")
    print(f"  {'order':<6} {'rel=-1 (out)':>13} {'rel=0 (in)':>12} {'delta':>9}")
    for order in ("A", "B", "C"):
        before = [float(x["auc"]) for x in long_rows
                  if x["order"] == order and x["rel_to_entry"] == -1 and x["auc"] != ""]
        after = [float(x["auc"]) for x in long_rows
                 if x["order"] == order and x["rel_to_entry"] == 0 and x["auc"] != ""]
        if before and after:
            mb, ma = sum(before) / len(before), sum(after) / len(after)
            print(f"  {order:<6} {mb:>13.4f} {ma:>12.4f} {ma - mb:>+9.4f}")
        else:
            print(f"  {order:<6} {'--':>13} {'--':>12} {'--':>9}")

    if missing:
        print(f"\n{len(missing)} row(s) with missing cells:", file=sys.stderr)
        for m in missing[:20]:
            print(f"  {m}", file=sys.stderr)
        if not args.allow_partial:
            print("\nexiting 1 (use --allow-partial to write anyway)", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
