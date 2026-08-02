#!/usr/bin/env python3
"""Resolve the newest complete checkpoint for each unique 2-hop ladder model."""

from __future__ import annotations

import argparse
from pathlib import Path

from make_configs import phase_rows


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]


def prefixes(phase: str) -> list[str]:
    if phase == "smoke":
        return ["nm_ladder_h2m_smoke_election"]
    return [str(row["prefix"]) for row in phase_rows(phase)]


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["smoke", "A", "robustness", "all"], default="A")
    parser.add_argument("--state-dir", type=Path, default=REPO_ROOT / "state")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    step = args.step if args.step is not None else (20 if args.phase == "smoke" else 40000)
    out = args.out or HERE / f"model_list_{args.phase}.txt"
    rows: list[str] = []
    missing: list[str] = []
    for prefix in prefixes(args.phase):
        checkpoint = complete_checkpoint(args.state_dir, prefix, step)
        if checkpoint is None:
            missing.append(prefix)
        else:
            rows.append(f"{prefix} {checkpoint}")

    out.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    print(f"wrote {out} ({len(rows)}/{len(rows) + len(missing)} models at step {step})")
    for prefix in missing:
        print(f"WARN: no complete step-{step} checkpoint for {prefix}")
    if missing and not args.allow_partial:
        return 1
    if not rows:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
