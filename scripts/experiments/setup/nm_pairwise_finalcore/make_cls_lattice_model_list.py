#!/usr/bin/env python3
"""Resolve the 9 specialist + 36 pair + 9 LOO step-2500 checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from scripts.experiments.setup.final_core.core_plan import SOURCES


DEFAULT_SPECIALIST_ROOT = Path(
    "/dataMeR1/phil/gfm/worktree-runtime-archive-20260812/"
    "prodigy-final-core/files/state/final_core"
)
DEFAULT_PAIR_ROOT = Path(
    "/dataMeR1/phil/gfm/prodigy-nm-pairs/log/nm_pairwise_finalcore/"
    "shared_seed0_20260904/state"
)
DEFAULT_LOO_ROOT = Path(
    "/dataMeR1/phil/gfm/prodigy-nm-loo/log/nm_leave_one_out_finalcore/"
    "shared_seed0_20260905_retry1/state"
)


def checkpoint_dirs(root: Path):
    return sorted(root.glob("*/checkpoint/state_dict_2500.ckpt"))


def build_rows(specialist_root=DEFAULT_SPECIALIST_ROOT, pair_root=DEFAULT_PAIR_ROOT, loo_root=DEFAULT_LOO_ROOT):
    rows = []
    for checkpoint in checkpoint_dirs(specialist_root):
        match = re.fullmatch(r"finalcore_ss_(.+)_s0_.+", checkpoint.parents[1].name)
        if match:
            source = match.group(1)
            rows.append((f"ss_{source}", checkpoint, (source,)))

    for checkpoint in checkpoint_dirs(pair_root):
        match = re.fullmatch(r"(nmpair_(.+)__(.+))_.+", checkpoint.parents[1].name)
        if match:
            rows.append((match.group(1), checkpoint, (match.group(2), match.group(3))))

    source_set = set(SOURCES)
    for checkpoint in checkpoint_dirs(loo_root):
        match = re.fullmatch(r"(nmloo_without_(.+))_.+", checkpoint.parents[1].name)
        if match:
            heldout = match.group(2)
            rows.append((match.group(1), checkpoint, tuple(source for source in SOURCES if source != heldout)))
            if heldout not in source_set:
                raise ValueError(f"unknown LOO heldout source: {heldout}")

    counts = {
        "specialist": sum(model_id.startswith("ss_") for model_id, _, _ in rows),
        "pair": sum(model_id.startswith("nmpair_") for model_id, _, _ in rows),
        "loo": sum(model_id.startswith("nmloo_") for model_id, _, _ in rows),
    }
    if counts != {"specialist": 9, "pair": 36, "loo": 9}:
        raise ValueError(f"checkpoint count mismatch: {counts}")
    ids = [model_id for model_id, _, _ in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate model ids")
    if any(not checkpoint.is_file() for _, checkpoint, _ in rows):
        raise FileNotFoundError("one or more resolved checkpoints disappeared")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--specialist-root", type=Path, default=DEFAULT_SPECIALIST_ROOT)
    parser.add_argument("--pair-root", type=Path, default=DEFAULT_PAIR_ROOT)
    parser.add_argument("--loo-root", type=Path, default=DEFAULT_LOO_ROOT)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("model_id\tcheckpoint\tsources\n")
        for model_id, checkpoint, sources in build_rows(args.specialist_root, args.pair_root, args.loo_root):
            handle.write(f"{model_id}\t{checkpoint}\t{','.join(sources)}\n")
    print(output)


if __name__ == "__main__":
    main()
