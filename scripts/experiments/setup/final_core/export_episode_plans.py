#!/usr/bin/env python3
"""Export final-core raw episode plans from the checked-out evaluator revision."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from core_plan import SOURCES
from evaluate_fixed_grid import (
    EPISODE_COUNT, fingerprint_plan, load_dataset, physical_jobs,
    reset_fixed_eval_rng, resolved_params, seed_everything, TrainerFS,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--expected-fingerprints", required=True, type=Path)
    args = parser.parse_args()
    targets = [value.strip() for value in args.targets.split(",") if value.strip()]
    expected = {}
    lines = args.expected_fingerprints.read_text().splitlines()
    header = lines[0].split("\t")
    for line in lines[1:]:
        row = dict(zip(header, line.split("\t")))
        expected[row["target"]] = row["episode_plan_fingerprint"]
    script_dir = Path(__file__).resolve().parent
    evaluator_args = SimpleNamespace(
        config=script_dir / "training.yaml", worker_index=0,
        evaluation_run_stamp="plan_export", evaluation_state_root=args.output_root / "state",
        evaluation_log_root=args.output_root / "log", batch_size=32, batch_count=16,
    )
    job = physical_jobs()[0]
    params = resolved_params(
        evaluator_args, seed=0, model_id=job.model.model_id,
        target=targets[0], checkpoint=args.checkpoint,
    )
    seed_everything(params)
    dataset = load_dataset(params)
    seed_everything(params)
    trainer = TrainerFS(dataset, params)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for target in targets:
        reset_fixed_eval_rng(target)
        trainer.parameter["neighbor_sampling_source_subset"] = target
        _, _, _, loader = trainer._build_dataloaders(dataset, trainer.dataset_name)
        batches = list(loader.batch_sampler)
        fingerprint, count = fingerprint_plan(
            target, batches, expected_batch_size=32, dataset=dataset
        )
        if count != EPISODE_COUNT or fingerprint != expected[target]:
            raise AssertionError(
                f"{target}: count={count} fingerprint={fingerprint} expected={expected[target]}"
            )
        torch.save(batches, args.output_root / f"{target}.pt")
        print(f"EXPORTED target={target} fingerprint={fingerprint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
