#!/usr/bin/env python3
"""Resolve the 6 dense saturation checkpoints (100 and 500 per arm) into a model list.

The other twelve rows of the 18-checkpoint curve come from
``setup/pretrain_saturation_existing``. Arm definitions and the model-key convention are
shared with that folder via ``arms.py`` -- imported by path, the same pattern
``nm_ladder_downstream/make_model_list.py`` uses -- so the two halves join in the
analysis instead of producing two disjoint sets of rows.

Usage (on Tucker, after run_all_train_tucker.sh):
    python3 make_model_list.py
    STATE_DIR=/some/other/state python3 make_model_list.py
    python3 make_model_list.py --dry-run

Exits nonzero if any checkpoint is missing: a hole at 100 or 500 is exactly the part of
the curve this experiment exists to measure.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXISTING = HERE.parent / "pretrain_saturation_existing"
# APPEND, never insert(0): the sibling folder has a same-named `make_model_list.py`, so
# putting it at the front of sys.path shadows this module for anything that imports it
# (check_splice.py does).
sys.path.append(str(EXISTING))

from arms import ARMS, DENSE_STEPS, default_dense_state_dir, model_key  # noqa: E402


def resolve_dense_run_dir(state_dir: Path, prefix: str) -> Path | None:
    """Newest ``<state_dir>/<prefix>_<timestamp>/`` directory.

    Matches on ``<prefix>_`` and prefers the most recent, so a re-run supersedes an
    earlier attempt rather than being picked at random.
    """
    candidates = sorted(
        (d for d in state_dir.glob(f"{prefix}_*") if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main() -> int:
    ap = argparse.ArgumentParser()
    # THIS worktree's state/, not the main checkout's: the dense runs wrote wherever
    # run_all_train_tucker.sh was launched from, and state/ does not follow a branch.
    ap.add_argument("--state-dir",
                    default=os.environ.get("DENSE_STATE_DIR", str(default_dense_state_dir())),
                    help="Where the DENSE retrains wrote their checkpoints "
                         "(default: this checkout's state/).")
    ap.add_argument("--out", default=str(HERE / "model_list.txt"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"{len(ARMS)} arms x {len(DENSE_STEPS)} steps "
          f"= {len(ARMS) * len(DENSE_STEPS)} checkpoints", flush=True)
    for arm in ARMS:
        print(f"  {arm.name:<6} prefix={arm.dense_prefix}  config={arm.dense_config}")
    if args.dry_run:
        return 0

    state_dir = Path(args.state_dir)
    if not state_dir.is_dir():
        print(f"ERROR: state dir not found: {state_dir}", file=sys.stderr)
        return 2

    rows: list[tuple[str, Path]] = []
    missing: list[str] = []
    for arm in ARMS:
        run_dir = resolve_dense_run_dir(state_dir, arm.dense_prefix)
        if run_dir is None:
            missing.append(f"{arm.name}: no run dir matching {arm.dense_prefix}_* "
                           f"under {state_dir} (has run_all_train_tucker.sh finished?)")
            continue
        print(f"  {arm.name:<6} -> {run_dir.name}")
        for step in DENSE_STEPS:
            ckpt = run_dir / "checkpoint" / f"state_dict_{step}.ckpt"
            if ckpt.is_file():
                rows.append((model_key(arm.name, step), ckpt))
            else:
                missing.append(f"{model_key(arm.name, step)}  {ckpt}")

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as fh:
        fh.write("# pretrain-saturation: densely-checkpointed early training.\n")
        fh.write("# Written by the post-2026-07-26 trainer: `state_dict_N` holds exactly\n")
        fh.write("# N completed optimizer steps. See README.md.\n")
        for key, ckpt in rows:
            fh.write(f"{key} {ckpt}\n")

    print(f"\nwrote {out} ({len(rows)}/{len(ARMS) * len(DENSE_STEPS)} checkpoints)")
    if missing:
        print(f"\nERROR: {len(missing)} checkpoint(s) missing:", file=sys.stderr)
        for row in missing:
            print(f"  {row}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
