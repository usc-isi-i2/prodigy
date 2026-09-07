#!/usr/bin/env python3
"""Resolve the 9 specialist + 36 pair + 9 LOO step-2500 checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from scripts.experiments.setup.final_core.core_plan import SOURCES


SPECIALIST_ROOT = Path(
    "/dataMeR1/phil/gfm/worktree-runtime-archive-20260812/"
    "prodigy-final-core/files/state/final_core"
)
PAIR_ROOT = Path(
    "/dataMeR1/phil/gfm/prodigy-nm-pairs/log/nm_pairwise_finalcore/"
    "shared_seed0_20260904/state"
)
LOO_ROOT = Path(
    "/dataMeR1/phil/gfm/prodigy-nm-loo/log/nm_leave_one_out_finalcore/"
    "shared_seed0_20260905_retry1/state"
)


def checkpoint_dirs(root: Path):
    return sorted(root.glob("*/checkpoint/state_dict_2500.ckpt"))


def build_rows():
    rows = []
    for checkpoint in checkpoint_dirs(SPECIALIST_ROOT):
        match = re.fullmatch(r"finalcore_ss_(.+)_s0_20260807", checkpoint.parents[1].name)
        if match:
            source = match.group(1)
            rows.append((f"ss_{source}", checkpoint, (source,)))

    for checkpoint in checkpoint_dirs(PAIR_ROOT):
        match = re.fullmatch(r"(nmpair_(.+)__(.+))_20260904_\d{6}_\d{3}", checkpoint.parents[1].name)
        if match:
            rows.append((match.group(1), checkpoint, (match.group(2), match.group(3))))

    source_set = set(SOURCES)
    for checkpoint in checkpoint_dirs(LOO_ROOT):
        match = re.fullmatch(r"(nmloo_without_(.+))_20260905_\d{6}_\d{3}", checkpoint.parents[1].name)
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
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("model_id\tcheckpoint\tsources\n")
        for model_id, checkpoint, sources in build_rows():
            handle.write(f"{model_id}\t{checkpoint}\t{','.join(sources)}\n")
    print(output)


if __name__ == "__main__":
    main()
