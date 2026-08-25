#!/usr/bin/env python3
"""Train a filtered/sharded plan while loading the all-eight graph only once."""

from __future__ import annotations

import argparse
import gc
from datetime import datetime
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from experiments.params import get_params  # noqa: E402
from experiments.run_single_experiment import load_dataset, seed_everything  # noqa: E402
from experiments.trainer import TrainerFS  # noqa: E402
from make_plan import TARGET_SUBSETS, rows, validate  # noqa: E402


def csv_set(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def complete_checkpoint(state_dir: Path, prefix: str) -> Path | None:
    candidates = sorted(
        (path for path in state_dir.glob(f"{prefix}_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in candidates:
        checkpoint = run_dir / "checkpoint" / "state_dict_2000.ckpt"
        if checkpoint.is_file():
            return checkpoint
    return None


def select_rows(
    targets: set[str], sizes: set[int], shard_index: int, num_shards: int, limit: int
) -> list[dict[str, object]]:
    selected = [
        row for row in rows()
        if (not targets or targets.intersection(row["heldout_targets"]))
        and (not sizes or row["mixture_size"] in sizes)
    ]
    selected = [row for index, row in enumerate(selected) if index % num_shards == shard_index]
    return selected[:limit] if limit else selected


def params_for(
    row: dict[str, object], device: int, seed: int, timestamp: str, state_dir: Path
):
    return get_params(
        [
            "--config", str(HERE / "base_train.yaml"),
            "--neighbor_sampling_source_subset", ",".join(row["donors"]),
            "--prefix", str(row["prefix"]),
            "--device", str(device),
            "--seed", str(seed),
            "--timestamp", timestamp,
            "--state_dir", str(state_dir),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets", default="",
        help="optional held-out targets; selects reusable models valid for any listed target",
    )
    parser.add_argument("--sizes", default="1,2,3,4")
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--state-dir", type=Path, default=REPO_ROOT / "state")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    validate()
    if args.device not in {2, 3}:
        parser.error("this project currently owns only Tucker GPUs 2 and 3")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    targets = csv_set(args.targets)
    unknown_targets = targets - set(TARGET_SUBSETS)
    if unknown_targets:
        parser.error(f"unknown targets: {sorted(unknown_targets)}")
    try:
        sizes = {int(value) for value in csv_set(args.sizes)}
    except ValueError as exc:
        parser.error(str(exc))
    if not sizes <= {1, 2, 3, 4}:
        parser.error("sizes must be drawn from 1,2,3,4")

    selected = select_rows(targets, sizes, args.shard_index, args.num_shards, args.limit)
    pending = []
    for row in selected:
        checkpoint = complete_checkpoint(args.state_dir, str(row["prefix"]))
        if checkpoint and not args.rerun_completed:
            print(f"SKIP complete {row['prefix']} -> {checkpoint}", flush=True)
        else:
            pending.append(row)
    print(
        f"selected={len(selected)} pending={len(pending)} shard="
        f"{args.shard_index}/{args.num_shards} device={args.device}",
        flush=True,
    )
    if not pending:
        return 0

    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    if args.dry_run:
        for row in pending:
            print(
                f"DRY {row['prefix']} k={row['mixture_size']} donors={','.join(row['donors'])} "
                f"heldout={','.join(row['heldout_targets'])}",
                flush=True,
            )
        return 0

    torch.set_num_threads(4)
    first_params = params_for(pending[0], args.device, args.seed, timestamp, args.state_dir)
    seed_everything(first_params)
    print("Loading shared all-eight graph once for this worker...", flush=True)
    dataset = load_dataset(first_params)

    for index, row in enumerate(pending, 1):
        params = params_for(row, args.device, args.seed, timestamp, args.state_dir)
        seed_everything(params)
        print(
            f"[run {index}/{len(pending)}] {row['prefix']} k={row['mixture_size']} "
            f"donors={','.join(row['donors'])} heldout={','.join(row['heldout_targets'])}",
            flush=True,
        )
        trainer = TrainerFS(dataset, params)
        trainer.train()
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
