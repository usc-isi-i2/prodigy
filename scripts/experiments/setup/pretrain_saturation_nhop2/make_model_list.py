#!/usr/bin/env python3
"""Resolve the complete fresh trajectories into one 19-checkpoint model list.

The three step-0 checkpoints must be byte-identical. When they are, the evaluator scores
one shared step-0 model instead of doing the same work three times.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from arms import ARMS, SHARED_STEP0_KEY, STEPS, TRAINED_STEPS, model_key


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_complete_run(state_dir: Path, prefix: str) -> Path | None:
    candidates = sorted(
        (path for path in state_dir.glob(f"{prefix}_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        checkpoint_dir = candidate / "checkpoint"
        if all((checkpoint_dir / f"state_dict_{step}.ckpt").is_file() for step in STEPS):
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-dir",
        default=os.environ.get("STATE_DIR", str(REPO_ROOT / "state")),
    )
    parser.add_argument("--out", default=str(HERE / "model_list.txt"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"{len(ARMS)} fresh arms x {len(STEPS)} checkpoints; "
          f"step 0 collapses to one shared row")
    for arm in ARMS:
        print(f"  {arm.name:<6} {arm.prefix:<14} {arm.corpus}")
    if args.dry_run:
        return 0

    state_dir = Path(args.state_dir)
    if not state_dir.is_dir():
        print(f"ERROR: state dir not found: {state_dir}", file=sys.stderr)
        return 2

    run_dirs: dict[str, Path] = {}
    missing: list[str] = []
    for arm in ARMS:
        run_dir = resolve_complete_run(state_dir, arm.prefix)
        if run_dir is None:
            missing.append(f"{arm.name}: no complete {arm.prefix}_* run under {state_dir}")
        else:
            run_dirs[arm.name] = run_dir
            print(f"  {arm.name:<6} -> {run_dir.name}")
    if missing:
        print("ERROR: incomplete trajectories:", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        return 1

    step0_paths = {
        arm.name: run_dirs[arm.name] / "checkpoint" / "state_dict_0.ckpt"
        for arm in ARMS
    }
    hashes = {arm: digest(path) for arm, path in step0_paths.items()}
    if len(set(hashes.values())) != 1:
        print("ERROR: step-0 checkpoints are not byte-identical:", file=sys.stderr)
        for arm, value in hashes.items():
            print(f"  {arm}: {value}", file=sys.stderr)
        return 1
    print(f"  shared step 0 sha256={next(iter(hashes.values()))[:16]}...")

    rows: list[tuple[str, Path]] = [
        (SHARED_STEP0_KEY, step0_paths[ARMS[0].name])
    ]
    for arm in ARMS:
        for step in TRAINED_STEPS:
            rows.append((
                model_key(arm.name, step),
                run_dirs[arm.name] / "checkpoint" / f"state_dict_{step}.ckpt",
            ))

    out = Path(args.out)
    out.write_text(
        "# n_hop=2 pretrain saturation: one shared step-0 + 18 trained checkpoints.\n"
        "# All state_dict_N files contain exactly N completed optimizer steps.\n"
        + "".join(f"{key} {path}\n" for key, path in rows),
        encoding="utf-8",
    )
    print(f"wrote {out} ({len(rows)} checkpoints)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
