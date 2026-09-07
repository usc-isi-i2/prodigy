#!/usr/bin/env python3
"""Resolve terminal checkpoints for the unconfined ladder."""

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lines = []
    missing = []
    for rung in range(1, 9):
        prefix = f"nm_ladder_unconf_h2m_r{rung}"
        candidates = list(args.state_dir.glob(f"{prefix}_*/checkpoint/state_dict_40000.ckpt"))
        if not candidates:
            missing.append(prefix)
            continue
        checkpoint = max(candidates, key=lambda path: path.stat().st_mtime)
        lines.append(f"{prefix} {checkpoint.resolve()}")
    if missing:
        raise SystemExit("missing terminal checkpoints: " + ", ".join(missing))
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
