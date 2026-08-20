#!/usr/bin/env python3
"""Verify p=1 global episodes stay inside the requested final-core rung."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))

from data.covid19_twitter import resolve_source_subset  # noqa: E402
from data.dataloader import NeighborTask  # noqa: E402
from experiments.params import get_params  # noqa: E402
from experiments.run_single_experiment import load_dataset  # noqa: E402


def profile(dataset, params, episodes, seed):
    graph_ids = dataset.graph.graph_id.detach().cpu().numpy()
    source_names = list(dataset.graph.source_graph_names)
    available = sorted(set(graph_ids.tolist()))
    active = resolve_source_subset(
        params["neighbor_sampling_source_subset"], available, source_names
    )
    strata = [np.where(graph_ids == graph_id)[0].tolist() for graph_id in sorted(active)]
    task = NeighborTask(
        dataset.neighbor_sampler,
        dataset.graph.num_nodes,
        "inout",
        sampling_strategy="strict",
        strata=strata,
        confine_to_single_stratum=True,
        stratum_weighting="proportional",
        cross_source_prob=1.0,
        filter_min_degree=True,
    )
    rng = random.Random(seed)
    torch.manual_seed(seed)
    source_counts = []
    observed = set()
    n_member = params["n_shots"] + params["n_query"]
    for _ in range(episodes):
        episode = task.sample(
            params["n_way"], n_member, params["n_shots"], params["n_query"], rng
        )
        center_sources = {int(value) for value in graph_ids[list(episode)].tolist()}
        if not center_sources <= active:
            raise AssertionError(f"inactive source leaked into episode: {center_sources - active}")
        observed.update(center_sources)
        source_counts.append(len(center_sources))
    return {
        "active_source_ids": sorted(active),
        "active_source_names": [source_names[index] for index in sorted(active)],
        "observed_source_ids": sorted(observed),
        "episodes": episodes,
        "mixed_episode_count": sum(count > 1 for count in source_counts),
        "sources_per_episode_min": min(source_counts),
        "sources_per_episode_mean": float(np.mean(source_counts)),
        "sources_per_episode_max": max(source_counts),
        "ready": all(count > 1 for count in source_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    resolved = [
        get_params(["--config", str(config), "--device", "123"])
        for config in args.config
    ]
    dataset = load_dataset(resolved[0])
    profiles = [
        {"config": str(config), **profile(dataset, params, args.episodes, seed=index)}
        for index, (config, params) in enumerate(zip(args.config, resolved))
    ]
    payload = {
        "protocol": "finalcore_global_active_union_probe_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profiles": profiles,
        "ready": all(result["ready"] for result in profiles),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
