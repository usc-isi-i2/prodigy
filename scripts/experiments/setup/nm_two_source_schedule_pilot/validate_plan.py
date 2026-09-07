#!/usr/bin/env python3
"""Validate the six-arm two-source schedule pilot without loading graph data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
EXPECTED_MODELS = {
    "cov_covpol_interleaved",
    "cov_then_covpol",
    "covpol_then_cov",
    "cov_cphk_interleaved",
    "cov_then_cphk",
    "cphk_then_cov",
}
EXPECTED_CONFIG = {
    "graph_filename": "ukr_rus_covid_midterm_all9_facebook_final_core_split_seed0.pt",
    "task_name": "neighbor_matching",
    "edge_view": "static_train",
    "target_edge_view": "static_test",
    "neighbor_matching_edge_split": True,
    "n_hop": 2,
    "neighbor_sampling_hop_sizes": "9,9",
    "neighbor_sampling_node_limit": 101,
    "n_way": 30,
    "n_shots": 3,
    "n_query": 4,
    "batch_size": 1,
    "dataset_len_cap": 2500,
    "seed": 0,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load((HERE / "training.yaml").read_text(encoding="utf-8"))
    for key, expected in EXPECTED_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(f"training.yaml: {key}={config.get(key)!r}, expected {expected!r}")

    with (HERE / "plan.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if {row["model_id"] for row in rows} != EXPECTED_MODELS or len(rows) != 6:
        raise ValueError("plan must contain exactly the six registered models")

    for row in rows:
        sources = row["sources"].split(",")
        if len(sources) != 2 or len(set(sources)) != 2:
            raise ValueError(f"{row['model_id']}: expected two unique sources")
        if row["schedule"] == "interleaved":
            if row["sequence"] or row["sequence_steps"]:
                raise ValueError(f"{row['model_id']}: interleaved arm has a blocked sequence")
        elif row["schedule"] == "sequential":
            if set(row["sequence"].split(",")) != set(sources):
                raise ValueError(f"{row['model_id']}: sequence does not match source set")
            steps = [int(value) for value in row["sequence_steps"].split(",")]
            if steps != [1250, 1250] or sum(steps) != config["dataset_len_cap"]:
                raise ValueError(f"{row['model_id']}: invalid sequential allocation")
        else:
            raise ValueError(f"{row['model_id']}: unknown schedule {row['schedule']!r}")

    if args.check_data:
        graph = Path(config["root"]) / config["graph_filename"]
        if not graph.is_file():
            raise FileNotFoundError(graph)

    print("two-source schedule pilot: 2 pairs x 3 schedules = 6 models")
    print("each model: seed 0, 2,500 updates, 1,250 expected/exact updates per source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
