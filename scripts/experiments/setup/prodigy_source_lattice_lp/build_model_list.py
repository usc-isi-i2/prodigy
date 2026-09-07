#!/usr/bin/env python3
"""Resolve the 54 seed-0 final-core PRODIGY checkpoints on Tucker."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


GROUPS = (
    ("pair", Path("/dataMeR1/phil/gfm/prodigy-nm-pairs/log/nm_pairwise_finalcore"), "nmpair_", 36),
    ("loo", Path("/dataMeR1/phil/gfm/prodigy-nm-loo/log/nm_leave_one_out_finalcore"), "nmloo_", 9),
    ("single", Path("/dataMeR1/phil/gfm/prodigy-roleexposure"), "finalcore_ss_", 9),
)


def newest_unique(root: Path, prefix: str) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for checkpoint in root.glob("**/checkpoint/state_dict_2500.ckpt"):
        run = checkpoint.parent.parent.name
        if not run.startswith(prefix):
            continue
        # Strip only the launcher timestamp; the semantic source name remains.
        model_id = re.sub(r"_20\d{6}[^/]*$", "", run)
        previous = found.get(model_id)
        if previous is None or checkpoint.stat().st_mtime > previous.stat().st_mtime:
            found[model_id] = checkpoint
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows: list[tuple[str, Path]] = []
    for kind, root, prefix, expected in GROUPS:
        group = newest_unique(root, prefix)
        if len(group) != expected:
            raise RuntimeError(f"{kind}: expected {expected} checkpoints, found {len(group)} beneath {root}")
        rows.extend(sorted(group.items()))
    if len(rows) != 54 or len({name for name, _ in rows}) != 54:
        raise RuntimeError("checkpoint manifest is not exactly 54 unique models")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{name}\t{path}\n" for name, path in rows))
    print(f"wrote {len(rows)} checkpoints to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
