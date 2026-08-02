#!/usr/bin/env python3
"""Resolve complete terminal checkpoints for the split-aware ladder."""

from __future__ import annotations

import argparse
from pathlib import Path

from make_configs import plan

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]


def complete_checkpoint(state_dir: Path, prefix: str, step: int) -> Path | None:
    candidates = sorted(
        (p for p in state_dir.glob(f"{prefix}_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in candidates:
        checkpoint = run_dir / "checkpoint" / f"state_dict_{step}.ckpt"
        if checkpoint.is_file():
            return checkpoint
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=REPO_ROOT / "state")
    parser.add_argument("--out", type=Path, default=HERE / "model_list.txt")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    entries = ([{"prefix": "nm_ladder_tts_h2m_smoke_election"}]
               if args.smoke else plan())
    step = 20 if args.smoke else 40000
    rows, missing = [], []
    for entry in entries:
        prefix = str(entry["prefix"])
        checkpoint = complete_checkpoint(args.state_dir, prefix, step)
        (rows if checkpoint else missing).append(
            f"{prefix} {checkpoint}" if checkpoint else prefix
        )
    args.out.write_text("\n".join(rows) + ("\n" if rows else ""))
    print(f"wrote {args.out}: {len(rows)}/{len(entries)} complete")
    for prefix in missing:
        print(f"WARN missing {prefix} at step {step}")
    return 0 if rows and (not missing or args.allow_partial) else 1


if __name__ == "__main__":
    raise SystemExit(main())
