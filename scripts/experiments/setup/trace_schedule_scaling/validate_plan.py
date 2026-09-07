#!/usr/bin/env python3
"""Fail-closed validation for TRACE schedule plans and Tucker inputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import yaml

from scripts.experiments.setup.trace_schedule_scaling.make_plan import build_plan


GRAPH = Path(
    "/dataMeR1/phil/data/merged/graphs/"
    "ukr_rus_covid_midterm_all9_facebook_final_core_split_seed0.pt"
)


def validate(
    total_steps: int, replay_block: int,
    rungs: tuple[int, ...], seeds: tuple[int, ...],
) -> None:
    arms = build_plan(
        total_steps=total_steps, replay_block=replay_block,
        rungs=rungs, seeds=seeds,
    )
    expected = len(rungs) * len(seeds) * 3
    if len(arms) != expected or len({arm.model_id for arm in arms}) != expected:
        raise ValueError("plan is incomplete or has duplicate model ids")
    groups = defaultdict(list)
    for arm in arms:
        groups[(arm.rung, arm.seed)].append(arm)
    for key, group in groups.items():
        counts = {arm.source_counts for arm in group}
        sources = {arm.sources for arm in group}
        if len(group) != 3 or len(counts) != 1 or len(sources) != 1:
            raise ValueError(f"unmatched schedule group {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=2500)
    parser.add_argument("--replay-block", type=int, default=100)
    parser.add_argument("--rungs", default="2,3,4")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()
    rungs = tuple(int(value) for value in args.rungs.split(","))
    seeds = tuple(int(value) for value in args.seeds.split(","))
    validate(args.total_steps, args.replay_block, rungs, seeds)
    config = yaml.safe_load(args.config.read_text())
    required = {
        "batch_size": 1,
        "epochs": 1,
        "neighbor_matching_edge_split": True,
        "neighbor_sampling_episode_source": "graph_id",
        "neighbor_sampling_cross_source_prob": 0.0,
        "neighbor_matching_member_policy": "uniform_shuffled",
        "train_episode_audit": True,
        "track_training_user_roles": False,
    }
    wrong = {key: (config.get(key), value) for key, value in required.items()
             if config.get(key) != value}
    if wrong:
        raise ValueError(f"training config violates the schedule contract: {wrong}")
    if args.check_data and not GRAPH.is_file():
        raise FileNotFoundError(GRAPH)
    print(
        f"OK: {len(rungs) * len(seeds) * 3} arms; order-only matched exposure; "
        f"total_steps={args.total_steps}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
