#!/usr/bin/env python3
"""Train one plan shard, or one exact model selected by prefix."""

from __future__ import annotations

import argparse
import gc
from datetime import datetime
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path[:0] = [str(REPO_ROOT), str(HERE)]

from experiments.params import get_params  # noqa: E402
from experiments.run_single_experiment import load_dataset, seed_everything  # noqa: E402
from experiments.trainer import TrainerFS  # noqa: E402
from make_plan import all_five_row, rows, validate  # noqa: E402


def complete(state_root: Path, prefix: str, step: int = 500) -> Path | None:
    for run_dir in sorted(state_root.glob(f"{prefix}_*"), key=lambda p: p.stat().st_mtime, reverse=True):
        checkpoint = run_dir / "checkpoint" / f"state_dict_{step}.ckpt"
        if checkpoint.is_file():
            return checkpoint
    return None


def training_state(state_root: Path, prefix: str, step: int = 500) -> Path | None:
    for run_dir in sorted(state_root.glob(f"{prefix}_*"), key=lambda p: p.stat().st_mtime, reverse=True):
        checkpoint = run_dir / "checkpoint" / f"training_state_{step}.ckpt"
        if checkpoint.is_file():
            return checkpoint
    return None


def resolved(row, args, stamp):
    return get_params([
        "--config", str(HERE / "train.yaml"),
        "--neighbor_sampling_source_subset", ",".join(row["donors"]),
        "--prefix", str(row["prefix"]), "--timestamp", stamp,
        "--device", str(args.device), "--state_dir", str(args.state_root),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--model-prefix", default="",
        help="train exactly one plan row; used to isolate DataLoader workers per model",
    )
    parser.add_argument("--state-root", type=Path, default=REPO_ROOT / "state")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.device not in {0, 1}:
        parser.error("only Tucker GPUs 0 and 1 are owned")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("bad shard")
    validate()
    if args.model_prefix:
        selected = [
            row for row in [*rows(), all_five_row()]
            if row["prefix"] == args.model_prefix
        ]
        if not selected:
            parser.error(f"unknown model prefix: {args.model_prefix}")
    else:
        selected = [
            row for index, row in enumerate(rows())
            if index % args.num_shards == args.shard_index
        ]
    pending = [row for row in selected if complete(args.state_root, str(row["prefix"])) is None]
    print(f"selected={len(selected)} pending={len(pending)} device={args.device}", flush=True)
    for row in selected:
        checkpoint = complete(args.state_root, str(row["prefix"]))
        if checkpoint:
            print(f"SKIP {row['prefix']} {checkpoint}", flush=True)
    if args.dry_run:
        for row in pending:
            print(f"DRY {row['prefix']} donors={','.join(row['donors'])}")
        return 0
    if not pending:
        return 0
    stamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    first = resolved(pending[0], args, stamp)
    seed_everything(first)
    print("Loading shared all-nine final-core graph once...", flush=True)
    dataset = load_dataset(first)
    for index, row in enumerate(pending, 1):
        params = resolved(row, args, stamp)
        seed_everything(params)
        print(f"[run {index}/{len(pending)}] {row['prefix']} donors={','.join(row['donors'])}", flush=True)
        trainer = TrainerFS(dataset, params)
        trainer.train()
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
