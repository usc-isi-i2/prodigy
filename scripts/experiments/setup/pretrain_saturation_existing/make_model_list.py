#!/usr/bin/env python3
"""Resolve the 12 already-trained saturation checkpoints and emit the sweep input.

Three arms x four steps (1000, 2000, 10000, 40000), all from checkpoint trajectories that
survived on Tucker -- no training happens for this half of the experiment. The other six
rows (100, 500) come from ``setup/pretrain_saturation_dense``.

Usage (on Tucker):
    python3 make_model_list.py
    STATE_DIR=/some/other/state python3 make_model_list.py
    python3 make_model_list.py --dry-run      # print the plan, resolve nothing

Exits nonzero if any checkpoint is missing: a partial model list silently produces a
curve with holes in it, which reads as a flat region rather than as missing data.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from arms import ARMS, DEFAULT_STATE_DIR, EXISTING_STEPS, model_key  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", default=os.environ.get("STATE_DIR", DEFAULT_STATE_DIR))
    ap.add_argument("--out", default=str(HERE / "model_list.txt"))
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and touch no disk.")
    args = ap.parse_args()

    print(f"{len(ARMS)} arms x {len(EXISTING_STEPS)} steps "
          f"= {len(ARMS) * len(EXISTING_STEPS)} checkpoints", flush=True)
    for arm in ARMS:
        print(f"  {arm.name:<6} {arm.run_dir}")

    if args.dry_run:
        for arm in ARMS:
            for step in EXISTING_STEPS:
                print(f"  {model_key(arm.name, step):<24} {arm.historical_ckpt(step, args.state_dir)}")
        return 0

    state_dir = Path(args.state_dir)
    if not state_dir.is_dir():
        print(f"ERROR: state dir not found: {state_dir}", file=sys.stderr)
        return 2

    rows: list[tuple[str, Path]] = []
    missing: list[str] = []
    for arm in ARMS:
        for step in EXISTING_STEPS:
            ckpt = arm.historical_ckpt(step, state_dir)
            if ckpt.is_file():
                rows.append((model_key(arm.name, step), ckpt))
            else:
                missing.append(f"{model_key(arm.name, step)}  {ckpt}")

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as fh:
        fh.write("# pretrain-saturation: checkpoints that already exist (no training).\n")
        fh.write("# NOTE: written by the pre-2026-07-26 trainer, so `state_dict_N` here\n")
        fh.write("# holds N+1 completed optimizer steps. See README.md.\n")
        for key, ckpt in rows:
            fh.write(f"{key} {ckpt}\n")

    print(f"\nwrote {out} ({len(rows)}/{len(ARMS) * len(EXISTING_STEPS)} checkpoints)")
    if missing:
        print(f"\nERROR: {len(missing)} checkpoint(s) missing under {state_dir}:",
              file=sys.stderr)
        for row in missing:
            print(f"  {row}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
