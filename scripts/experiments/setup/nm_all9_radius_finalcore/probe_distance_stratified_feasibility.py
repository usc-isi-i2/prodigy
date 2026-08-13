#!/usr/bin/env python3
"""Read-only feasibility gate for within-episode distance-stratified NM."""

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

from data.covid19_twitter import (  # noqa: E402
    _parse_center_distance_weights,
    _parse_positive_int_list,
)
from data.dataloader import NeighborTask  # noqa: E402
from experiments.params import get_params  # noqa: E402
from experiments.run_single_experiment import load_dataset  # noqa: E402


def percentile(values, q):
    return float(np.quantile(np.asarray(values, dtype=float), q)) if values else None


def profile_split(dataset, params, split, episodes, seed):
    positive_sampler = {
        "train": dataset.neighbor_sampler,
        "val": dataset.nm_validation_neighbor_sampler,
        "test": dataset.nm_test_neighbor_sampler,
    }[split]
    radii = _parse_positive_int_list(
        params["neighbor_sampling_center_distance_radii"],
        "neighbor_sampling_center_distance_radii",
    )
    weights = _parse_center_distance_weights(
        params["neighbor_sampling_center_distance_weights"], len(radii) + 1
    )
    task = NeighborTask(
        positive_sampler,
        dataset.graph.num_nodes,
        "inout",
        sampling_strategy="strict",
        filter_min_degree=True,
        center_distance_radii=radii,
        center_distance_weights=weights,
        center_region_fanout=params["neighbor_sampling_center_region_fanout"],
        center_region_node_limit=params["neighbor_sampling_center_region_node_limit"],
        center_region_candidate_limit=params[
            "neighbor_sampling_center_region_candidate_limit"
        ],
        center_region_sampler=dataset.neighbor_sampler,
        center_max_attempts=params["neighbor_sampling_center_max_attempts"],
    )
    expected_counts = task._distance_band_counts(params["n_way"])
    rng = random.Random(seed)
    torch.manual_seed(seed)
    graph_ids = getattr(dataset.graph, "graph_id", None)
    n_member = params["n_shots"] + params["n_query"]
    attempts = []
    elapsed = []
    cross_source_global = 0
    failures = []

    for episode_idx in range(episodes):
        started = time.monotonic()
        try:
            episode = task.sample(
                params["n_way"], n_member, params["n_shots"], params["n_query"], rng
            )
        except RuntimeError as exc:
            failures.append({"episode": episode_idx, "error": str(exc)})
            continue
        elapsed.append(time.monotonic() - started)
        attempts.append(int(task.last_center_sampling_attempts))
        groups = task.last_sampled_center_distance_groups
        if [len(group) for group in groups] != expected_counts:
            raise AssertionError(f"wrong band counts: {groups}")
        centers = list(episode)
        members = [node for values in episode.values() for node in values]
        if len(centers) != params["n_way"] or len(centers) != len(set(centers)):
            raise AssertionError("wrong or duplicate centers")
        if len(members) != len(set(members)) or set(centers).intersection(members):
            raise AssertionError("episode contains an ambiguous node collision")
        if graph_ids is not None:
            anchor_source = int(graph_ids[groups[0][0]].item())
            finite_sources = {
                int(value)
                for value in graph_ids[
                    [node for group in groups[:-1] for node in group]
                ].tolist()
            }
            if finite_sources != {anchor_source}:
                raise AssertionError("finite distance bands crossed source components")
            global_sources = {
                int(value) for value in graph_ids[groups[-1]].tolist()
            }
            if any(source != anchor_source for source in global_sources):
                cross_source_global += 1

    ready = len(attempts) == episodes
    if graph_ids is not None:
        ready = ready and cross_source_global > 0
    return {
        "split": split,
        "requested_episodes": episodes,
        "successful_episodes": len(attempts),
        "band_radii": radii,
        "band_weights": weights,
        "resolved_counts": expected_counts,
        "failures": failures[:10],
        "attempts_mean": float(np.mean(attempts)) if attempts else None,
        "attempts_p95": percentile(attempts, 0.95),
        "attempts_max": max(attempts) if attempts else None,
        "seconds_mean": float(np.mean(elapsed)) if elapsed else None,
        "seconds_p95": percentile(elapsed, 0.95),
        "episodes_with_cross_source_global_band": cross_source_global,
        "ready": ready,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "distance_stratified.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    params = get_params(["--config", str(args.config), "--seed", str(args.seed), "--device", "123"])
    dataset = load_dataset(params)
    profiles = [
        profile_split(dataset, params, split, args.episodes, args.seed + index * 10)
        for index, split in enumerate(("train", "val", "test"))
    ]
    payload = {
        "protocol": "nm_all9_distance_stratified_feasibility_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "episodes_per_split": args.episodes,
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
