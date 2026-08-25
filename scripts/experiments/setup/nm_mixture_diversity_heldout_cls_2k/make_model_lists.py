#!/usr/bin/env python3
"""Resolve completed 2k checkpoints into one model list per held-out target."""

from __future__ import annotations

import argparse
from pathlib import Path

from make_plan import TARGET_SUBSETS, evaluation_rows
from run_sweep import complete_checkpoint


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=REPO_ROOT / "state")
    parser.add_argument("--out-dir", type=Path, default=HERE / "model_lists")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    missing_total = 0
    for target in TARGET_SUBSETS:
        target_rows = [row for row in evaluation_rows() if row["target"] == target]
        resolved = []
        missing = []
        for row in target_rows:
            checkpoint = complete_checkpoint(args.state_dir, str(row["prefix"]))
            if checkpoint is None:
                missing.append(str(row["prefix"]))
            else:
                resolved.append(f"{row['prefix']} {checkpoint}")
        out = args.out_dir / f"{target}.txt"
        out.write_text("\n".join(resolved) + ("\n" if resolved else ""), encoding="utf-8")
        print(f"{target}: {len(resolved)}/{len(target_rows)} -> {out}")
        for prefix in missing:
            print(f"WARN missing step-2000 checkpoint: {prefix}")
        missing_total += len(missing)
    return int(missing_total > 0 and not args.allow_partial)


if __name__ == "__main__":
    raise SystemExit(main())
