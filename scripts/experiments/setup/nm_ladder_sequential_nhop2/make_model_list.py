#!/usr/bin/env python3
"""Resolve complete sequential-ladder checkpoints into an eval model list."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from make_configs import plan


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]


def resolve_checkpoint(state_dir: Path, prefix: str, step: int) -> Path | None:
    candidates = []
    for run_dir in state_dir.glob(f"{prefix}_*"):
        checkpoint = run_dir / "checkpoint" / f"state_dict_{step}.ckpt"
        if checkpoint.is_file():
            candidates.append((run_dir.stat().st_mtime, checkpoint))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=REPO_ROOT / "state")
    parser.add_argument("--output", type=Path, default=HERE / "model_list.txt")
    parser.add_argument("--phase", choices=["all", "smoke"], default="all")
    args = parser.parse_args()

    if args.phase == "smoke":
        requested = [("nm_ladder_seq_h2m_smoke", 20)]
    else:
        requested = [(str(row["prefix"]), 40_000) for row in plan()]

    lines = []
    missing = []
    for prefix, step in requested:
        checkpoint = resolve_checkpoint(args.state_dir, prefix, step)
        if checkpoint is None:
            missing.append(f"{prefix}: state_dict_{step}.ckpt")
        else:
            lines.append(f"{prefix} {checkpoint.resolve()}")

    if missing:
        print("missing complete checkpoints:", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        return 1

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(lines)} checkpoints)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
