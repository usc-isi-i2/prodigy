#!/usr/bin/env python3
"""Resolve each fixed-exposure ladder model at its rung-specific final step."""

from __future__ import annotations

import argparse
from pathlib import Path

from make_configs import phase_rows


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]


def requested_models(phase: str) -> list[tuple[str, int]]:
    if phase == "smoke":
        return [("nm_ladder_fx10k_h2m_smoke_election", 20)]
    return [
        (str(row["prefix"]), int(row["target_step"]))
        for row in phase_rows(phase)
    ]


def complete_checkpoint(state_dir: Path, prefix: str, step: int) -> Path | None:
    candidates = sorted(
        (path for path in state_dir.glob(f"{prefix}_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in candidates:
        checkpoint = run_dir / "checkpoint" / f"state_dict_{step}.ckpt"
        if checkpoint.is_file():
            return checkpoint
    return None


def resolve_models(
    state_dir: Path, phase: str
) -> tuple[list[tuple[str, int, Path]], list[tuple[str, int]]]:
    resolved: list[tuple[str, int, Path]] = []
    missing: list[tuple[str, int]] = []
    for prefix, step in requested_models(phase):
        checkpoint = complete_checkpoint(state_dir, prefix, step)
        if checkpoint is None:
            missing.append((prefix, step))
        else:
            resolved.append((prefix, step, checkpoint))
    return resolved, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["smoke", "A", "robustness", "all"], default="A")
    parser.add_argument("--state-dir", type=Path, default=REPO_ROOT / "state")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    out = args.out or HERE / f"model_list_{args.phase}.txt"
    resolved, missing = resolve_models(args.state_dir, args.phase)
    lines = [f"{prefix} {checkpoint}" for prefix, _, checkpoint in resolved]
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"wrote {out} ({len(resolved)}/{len(resolved) + len(missing)} models)")
    for prefix, step in missing:
        print(f"WARN: no complete step-{step} checkpoint for {prefix}")
    if missing and not args.allow_partial:
        return 1
    if not resolved:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
