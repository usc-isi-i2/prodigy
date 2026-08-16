#!/usr/bin/env python3
"""Continue one mixture from step 500 to 1,000 with a fresh episode stream."""

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
from run_train import complete, training_state  # noqa: E402

START_STEP = 500
CONTINUATION_STEPS = 500
CHECKPOINT_STEPS = "250,500"
CONTINUATION_SEED = 500


def continuation_prefix(prefix: str) -> str:
    if not prefix.startswith("labmix500_"):
        raise ValueError(f"unexpected original prefix: {prefix}")
    return prefix.replace("labmix500_", "labmixcont_", 1)


def resolved(row, args, stamp):
    return get_params([
        "--config", str(HERE / "train.yaml"),
        "--neighbor_sampling_source_subset", ",".join(row["donors"]),
        "--prefix", continuation_prefix(str(row["prefix"])), "--timestamp", stamp,
        "--device", str(args.device), "--state_dir", str(args.continuation_state_root),
        "--dataset_len_cap", str(CONTINUATION_STEPS),
        "--checkpoint_step", "250", "--checkpoint_steps", CHECKPOINT_STEPS,
        "--seed", str(CONTINUATION_SEED), "--workers", "2",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--model-prefix", required=True)
    parser.add_argument("--source-state-root", type=Path, default=REPO_ROOT / "state")
    parser.add_argument(
        "--continuation-state-root", type=Path,
        default=REPO_ROOT / "state_labmix500_continuation",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.device not in {0, 1}:
        parser.error("only Tucker GPUs 0 and 1 are owned")
    validate()
    selected = [
        row for row in [*rows(), all_five_row()]
        if row["prefix"] == args.model_prefix
    ]
    if not selected:
        parser.error(f"unknown model prefix: {args.model_prefix}")
    row = selected[0]
    output_prefix = continuation_prefix(str(row["prefix"]))
    source = training_state(args.source_state_root, str(row["prefix"]), START_STEP)
    if source is None:
        raise FileNotFoundError(f"missing step-{START_STEP} training state for {row['prefix']}")
    terminal = complete(args.continuation_state_root, output_prefix, CONTINUATION_STEPS)
    if terminal:
        print(f"SKIP {row['prefix']} continuation complete: {terminal}", flush=True)
        return 0
    print(
        f"CONTINUE {row['prefix']} source={source} local_steps={CONTINUATION_STEPS} "
        f"global_checkpoints=750,1000 fresh_episode_seed={CONTINUATION_SEED}",
        flush=True,
    )
    if args.dry_run:
        return 0

    stamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    params = resolved(row, args, stamp)
    seed_everything(params)
    dataset = load_dataset(params)
    trainer = TrainerFS(dataset, params)
    payload = trainer._torch_load_checkpoint(source, map_location=trainer.device)
    training = payload.get("_training_checkpoint")
    if training is None or "optimizer" not in training:
        raise ValueError(f"checkpoint lacks optimizer state: {source}")
    if int(training.get("completed_steps", -1)) != START_STEP:
        raise ValueError(f"checkpoint is not at step {START_STEP}: {source}")
    trainer.load_checkpoint(source)
    trainer.optimizer.load_state_dict(training["optimizer"])
    # Prefetch makes the exact next episode unknowable for 21 original runs. Reset
    # every arm identically instead of treating zero-worker and two-worker runs differently.
    seed_everything(params)
    trainer.train()
    del trainer, dataset, payload, training
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
