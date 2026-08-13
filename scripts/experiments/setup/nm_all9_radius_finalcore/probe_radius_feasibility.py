#!/usr/bin/env python3
"""Read-only Tucker feasibility gate for 30-way radius-confined NM episodes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))

from data.dataloader import NeighborTask  # noqa: E402
from experiments.params import get_params  # noqa: E402
from experiments.run_single_experiment import load_dataset  # noqa: E402


def percentile(values, q):
    return float(np.quantile(np.asarray(values, dtype=float), q)) if values else None


def profile_radius(dataset, params, split, radius, episodes, seed):
    if split == "train":
        positive_sampler = dataset.neighbor_sampler
    elif split == "val":
        positive_sampler = dataset.nm_validation_neighbor_sampler
    elif split == "test":
        positive_sampler = dataset.nm_test_neighbor_sampler
    else:
        raise ValueError(f"unknown split {split!r}")
    task = NeighborTask(
        positive_sampler,
        dataset.graph.num_nodes,
        "inout",
        sampling_strategy="strict",
        filter_min_degree=True,
        center_radii=[radius],
        center_radius_weights=[1.0],
        center_region_fanout=params["neighbor_sampling_center_region_fanout"],
        center_region_node_limit=params["neighbor_sampling_center_region_node_limit"],
        center_region_candidate_limit=params[
            "neighbor_sampling_center_region_candidate_limit"
        ],
        center_region_sampler=dataset.neighbor_sampler,
        center_max_attempts=params["neighbor_sampling_center_max_attempts"],
    )
    rng = random.Random(seed)
    torch.manual_seed(seed)
    attempts = []
    source_counts = []
    elapsed = []
    failures = []
    graph_ids = getattr(dataset.graph, "graph_id", None)
    n_member = params["n_shots"] + params["n_query"]

    for episode_idx in range(episodes):
        started = time.monotonic()
        try:
            episode = task.sample(
                params["n_way"],
                n_member,
                params["n_shots"],
                params["n_query"],
                rng,
            )
        except RuntimeError as exc:
            failures.append({"episode": episode_idx, "error": str(exc)})
            continue
        elapsed.append(time.monotonic() - started)
        attempts.append(int(task.last_center_sampling_attempts))
        centers = list(episode)
        members = [node for values in episode.values() for node in values]
        if len(centers) != params["n_way"]:
            raise AssertionError("wrong number of centers")
        if len(members) != len(set(members)) or set(centers).intersection(members):
            raise AssertionError("episode contains an ambiguous cross-label node collision")
        if graph_ids is not None:
            source_counts.append(
                len(set(int(value) for value in graph_ids[centers].tolist()))
            )

    radius_text = "global" if radius is None else str(radius)
    record = {
        "split": split,
        "radius": radius_text,
        "requested_episodes": episodes,
        "successful_episodes": len(attempts),
        "failures": failures[:10],
        "attempts_mean": float(np.mean(attempts)) if attempts else None,
        "attempts_p50": percentile(attempts, 0.50),
        "attempts_p95": percentile(attempts, 0.95),
        "attempts_max": max(attempts) if attempts else None,
        "seconds_mean": float(np.mean(elapsed)) if elapsed else None,
        "seconds_p95": percentile(elapsed, 0.95),
        "sources_per_episode_mean": (
            float(np.mean(source_counts)) if source_counts else None
        ),
        "sources_per_episode_min": min(source_counts) if source_counts else None,
        "sources_per_episode_max": max(source_counts) if source_counts else None,
    }
    ready = len(attempts) == episodes
    if radius is not None and source_counts:
        ready = ready and max(source_counts) == 1
    if radius is None and source_counts:
        ready = ready and max(source_counts) > 1
    record["ready"] = ready
    return record


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "radius_mix.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    params = get_params(
        ["--config", str(args.config), "--seed", str(args.seed), "--device", "123"]
    )
    dataset = load_dataset(params)
    profiles = []
    for split_index, split in enumerate(("train", "val", "test")):
        for radius_index, radius in enumerate((2, 3, None)):
            profiles.append(
                profile_radius(
                    dataset,
                    params,
                    split,
                    radius,
                    args.episodes,
                    args.seed + split_index * 10 + radius_index,
                )
            )
    payload = {
        "protocol": "nm_all9_radius_finalcore_feasibility_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "episodes_per_radius": args.episodes,
        "seed": args.seed,
        "profiles": profiles,
        "ready": all(profile["ready"] for profile in profiles),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
