#!/usr/bin/env python3
"""Verify stage-1 terminal checkpoints, contracts, and consumed episodes."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import torch

from scripts.experiments.setup.final_core.core_plan import SOURCES as ALL_SOURCES
from scripts.experiments.setup.trace_health_guided.make_stage1_plan import build_plan


def find_audit(log_root: Path, run_name: str) -> Path:
    matches = [
        path
        for path in log_root.rglob("consumed_episodes.jsonl.gz")
        if run_name in path.as_posix()
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one audit for {run_name}, got {matches}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--total-steps", type=int, default=2500)
    parser.add_argument("--sources", default="ukr_rus,covid,midterm,covid_political")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = tuple(value for value in args.sources.split(",") if value)
    seeds = tuple(int(value) for value in args.seeds.split(","))
    receipts = []
    for arm in build_plan(sources=sources, seeds=seeds):
        run_name = f"tracehg_{arm.model_id}_{args.run_stamp}"
        checkpoint_root = args.state_root / run_name / "checkpoint"
        checkpoint = checkpoint_root / f"state_dict_{args.total_steps}.ckpt"
        sidecar = checkpoint_root / f"training_state_{args.total_steps}.ckpt"
        if not checkpoint.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"incomplete terminal checkpoint: {run_name}")
        training = torch.load(sidecar, map_location="cpu", weights_only=False)
        metadata = training.get("_training_checkpoint", {})
        if metadata.get("completed_steps") != args.total_steps:
            raise ValueError(f"wrong completed step for {run_name}")
        contract = metadata.get("parameter_contract", {})
        expected = {
            "seed": arm.seed,
            "neighbor_sampling_source_subset": ",".join(arm.sources),
            "neighbor_sampling_strata": "graph_id_pool",
            "neighbor_sampling_episode_source": (
                "graph_id" if arm.condition == "single" else ""
            ),
            "neighbor_matching_member_policy": "uniform_shuffled",
            "neighbor_matching_member_seed": 380100 + arm.seed,
            "dataset_len_cap": args.total_steps,
            "batch_size": 1,
        }
        if arm.condition == "single":
            expected.update(
                {
                    "neighbor_sampling_source_schedule": arm.sources[0],
                    "neighbor_sampling_source_schedule_steps": str(args.total_steps),
                    "neighbor_sampling_source_schedule_seed": 480100 + arm.seed,
                }
            )
        else:
            expected.update(
                {
                    "neighbor_sampling_source_schedule": "",
                    "neighbor_sampling_source_schedule_steps": "",
                    "neighbor_sampling_source_schedule_seed": -1,
                }
            )
        wrong = {
            key: (contract.get(key), value)
            for key, value in expected.items()
            if contract.get(key) != value
        }
        if wrong:
            raise ValueError(f"training contract differs for {run_name}: {wrong}")
        audit = find_audit(args.log_root, run_name)
        with gzip.open(audit, "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        if len(rows) != args.total_steps or [row["step"] for row in rows] != list(
            range(1, args.total_steps + 1)
        ):
            raise ValueError(f"incomplete consumed audit: {run_name}")
        if any(row["query_roles"] != [0, 0, 0, 1, 1, 1, 1] for row in rows):
            raise ValueError(f"unexpected support/query grid: {run_name}")
        if arm.condition == "single":
            expected_source = ALL_SOURCES.index(arm.sources[0])
            if any(row["source_ids"] != [expected_source] for row in rows):
                raise ValueError(f"single-source contamination: {run_name}")
            mixed_fraction = 0.0
        else:
            mixed_fraction = sum(row["source_ids"] == [-1] for row in rows) / len(rows)
            if mixed_fraction < 0.99:
                raise ValueError(
                    f"merged arm did not produce cross-source episodes: "
                    f"{run_name}, mixed_fraction={mixed_fraction:.6f}"
                )
        receipts.append(
            {
                "model_id": arm.model_id,
                "condition": arm.condition,
                "seed": arm.seed,
                "checkpoint": str(checkpoint),
                "audit": str(audit),
                "episodes": len(rows),
                "mixed_source_episode_fraction": mixed_fraction,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "complete",
                "arms": len(receipts),
                "total_steps": args.total_steps,
                "receipts": receipts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"OK: verified {len(receipts)} stage-1 arms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
