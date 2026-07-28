#!/usr/bin/env python3
"""Resolve every ladder rung to its matched-40k checkpoint and emit the sweep inputs.

The 24 table rows (3 orders x 8 rungs) are served by only **21 distinct encoders** --
rung 8 is order-invariant, order B's rung 2 has the same source SET as order A's rung 2,
and rung 1 of any order is that graph's single-source specialist. This script reuses
``nm_ladder_order_robustness-jul_23/make_configs.plan()`` so the new-vs-reuse mapping is
the same one the NM table was built from, rather than a second hand-maintained copy.

Emits two files next to this script:

  model_list.txt   one ``<model_key> <checkpoint>`` line per DISTINCT encoder (21).
                   Fed to the eval runner and to ``pair_link_sweep.py``.
  row_map.csv      24 rows: order, rung, added, sources, model_key. The assembler
                   expands the 21 encoders back into 24 rows through this file.

``model_key`` is the training run's own prefix (``nm_ladder_4src``,
``nm_ladder_ordB_r3``, ``nm_ss_covid19_twitter``, ...), not an invented alias, so a row
in the shared per-task CSVs can be traced straight back to a state dir.

Usage:
    STATE_DIR=/dataMeR1/phil/gfm/prodigy/state python3 make_model_list.py
    STATE_DIR=... STEP=40000 python3 make_model_list.py
    python3 make_model_list.py --dry-run      # print the plan, resolve nothing

Exits nonzero if any encoder is missing at ``STEP`` -- a partial sweep silently
mislabels rows, so the caller must not proceed on a warning.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER_ROB = HERE.parent / "nm_ladder_order_robustness-jul_23"
sys.path.insert(0, str(ORDER_ROB))

from make_configs import ORDERS, SOURCES, canonical, plan  # noqa: E402

DATASET_OF_KEY = {s[0]: s[1] for s in SOURCES}


def run_prefix(row: dict) -> str:
    """The training-run prefix backing this rung.

    ``plan()`` records reused rungs as a provenance sentence
    (``"order-A ladder: nm_ladder_4src"``); new rungs carry the bare prefix.
    """
    return row["run"].split(": ")[-1].strip()


def resolve_checkpoint(state_dir: Path, prefix: str, step: int) -> Path | None:
    """Newest ``<state_dir>/<prefix>_<timestamp>/checkpoint/state_dict_<step>.ckpt``.

    Matches on ``<prefix>_`` so ``ukr_only_nm`` cannot pick up ``ukr_only_nm_aug``,
    and prefers the most recent run dir when a prefix was trained more than once.
    """
    candidates = sorted(
        (d for d in state_dir.glob(f"{prefix}_*") if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for d in candidates:
        ckpt = d / "checkpoint" / f"state_dict_{step}.ckpt"
        if ckpt.is_file():
            return ckpt
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default=os.environ.get(
        "STATE_DIR", "/dataMeR1/phil/gfm/prodigy/state"))
    ap.add_argument("--step", type=int, default=int(os.environ.get("STEP", "40000")))
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the 24-row plan and the distinct encoder set; touch no disk.")
    args = ap.parse_args()

    rows = plan()
    assert len(rows) == 24, f"expected 24 (order, rung) rows, got {len(rows)}"

    for row in rows:
        row["model_key"] = run_prefix(row)

    # Distinct encoders, in first-appearance order so model_list.txt reads like the table.
    seen: dict[str, None] = {}
    for row in rows:
        seen.setdefault(row["model_key"], None)
    keys = list(seen)

    print(f"{len(rows)} rows -> {len(keys)} distinct encoders", flush=True)
    for row in rows:
        srcs = " ".join(row["sources"])
        print(f"  {row['order']}{row['rung']}  +{row['added']:<18} "
              f"[{row['status']:>5}] {row['model_key']:<40} {srcs}")

    if args.dry_run:
        return 0

    state_dir = Path(args.state_dir)
    if not state_dir.is_dir():
        print(f"ERROR: state dir not found: {state_dir}", file=sys.stderr)
        return 2

    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for key in keys:
        ckpt = resolve_checkpoint(state_dir, key, args.step)
        if ckpt is None:
            missing.append(key)
        else:
            resolved[key] = ckpt

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_list = out_dir / "model_list.txt"
    with model_list.open("w", encoding="utf-8") as fh:
        for key in keys:
            if key in resolved:
                fh.write(f"{key} {resolved[key]}\n")

    row_map = out_dir / "row_map.csv"
    with row_map.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["order", "rung", "added", "added_canonical", "added_dataset",
                    "n_sources", "sources", "status", "model_key"])
        for row in rows:
            w.writerow([
                row["order"], row["rung"], row["added"], canonical(row["added"]),
                DATASET_OF_KEY[row["added"]], len(row["sources"]),
                " ".join(row["sources"]), row["status"], row["model_key"],
            ])

    print(f"\nwrote {model_list} ({len(resolved)}/{len(keys)} encoders)")
    print(f"wrote {row_map} (24 rows)")

    if missing:
        print(f"\nERROR: no state_dict_{args.step}.ckpt for {len(missing)} encoder(s) "
              f"under {state_dir}:", file=sys.stderr)
        for key in missing:
            print(f"  {key}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
