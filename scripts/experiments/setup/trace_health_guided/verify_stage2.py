#!/usr/bin/env python3
"""Verify health-guided terminal checkpoints and exact realized schedules."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import torch

from scripts.experiments.setup.final_core.core_plan import SOURCES as ALL_SOURCES


def expand(sources: str, steps: str) -> list[str]:
    return [
        source
        for source, count in zip(sources.split(","), map(int, steps.split(",")))
        for _ in range(count)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--total-steps", type=int, default=2500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.plan.open(newline="", encoding="utf-8") as handle:
        plan = list(csv.DictReader(handle, delimiter="\t"))
    if len(plan) != 3:
        raise ValueError(f"expected three health-guided arms, got {len(plan)}")
    receipts = []
    for arm in plan:
        seed = int(arm["seed"])
        run_name = f"tracehg_{arm['model_id']}_{args.run_stamp}"
        checkpoint_root = args.state_root / run_name / "checkpoint"
        checkpoint = checkpoint_root / f"state_dict_{args.total_steps}.ckpt"
        sidecar = checkpoint_root / f"training_state_{args.total_steps}.ckpt"
        if not checkpoint.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"incomplete health-guided checkpoint: {run_name}")
        training = torch.load(sidecar, map_location="cpu", weights_only=False)
        metadata = training.get("_training_checkpoint", {})
        contract = metadata.get("parameter_contract", {})
        expected = {
            "seed": seed,
            "dataset_len_cap": args.total_steps,
            "neighbor_sampling_source_subset": arm["sources"],
            "neighbor_sampling_episode_source": "graph_id",
            "neighbor_sampling_source_schedule": arm["segment_sources"],
            "neighbor_sampling_source_schedule_steps": arm["segment_steps"],
            "neighbor_sampling_source_schedule_seed": 480100 + seed,
            "neighbor_matching_member_seed": 380100 + seed,
        }
        wrong = {
            key: (contract.get(key), value)
            for key, value in expected.items()
            if contract.get(key) != value
        }
        if metadata.get("completed_steps") != args.total_steps or wrong:
            raise ValueError(f"health-guided contract differs for {run_name}: {wrong}")
        matches = [
            path
            for path in args.log_root.rglob("consumed_episodes.jsonl.gz")
            if run_name in path.as_posix()
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one consumed audit for {run_name}")
        with gzip.open(matches[0], "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        expected_sources = expand(arm["segment_sources"], arm["segment_steps"])
        if len(rows) != args.total_steps or len(expected_sources) != args.total_steps:
            raise ValueError(f"incomplete health-guided audit: {run_name}")
        for row, source in zip(rows, expected_sources):
            if row["source_ids"] != [ALL_SOURCES.index(source)]:
                raise ValueError(
                    f"realized source differs at {run_name} step {row['step']}"
                )
        receipts.append(
            {
                "model_id": arm["model_id"],
                "checkpoint": str(checkpoint),
                "audit": str(matches[0]),
                "episodes": len(rows),
                "ranking": arm["ranked_sources"],
                "rank_counts": arm["rank_counts"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"status": "complete", "arms": len(receipts), "receipts": receipts},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"OK: verified {len(receipts)} health-guided schedules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
