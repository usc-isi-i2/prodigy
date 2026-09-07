#!/usr/bin/env python3
"""Validate the registered one-seed social-specificity pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCES = ("ukr_rus_twitter", "facebook_page_reference", "cora", "pubmed")
TARGETS = SOURCES
EXPECTED = {
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
    "batch_size": 4,
    "dataset_len_cap": 2500,
    "seed": 0,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-data", action="store_true")
    args = parser.parse_args()

    catalog = json.loads((ROOT / "docs/graph_catalog.json").read_text(encoding="utf-8"))
    by_key = {row["dataset_key"]: row for row in catalog["graphs"]}
    missing_catalog = sorted(set(TARGETS) - set(by_key))
    if missing_catalog:
        raise ValueError(f"graphs absent from catalog: {missing_catalog}")

    for dataset in ("cora", "pubmed"):
        config_path = HERE / f"{dataset}_nm.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for key, expected in EXPECTED.items():
            if config.get(key) != expected:
                raise ValueError(
                    f"{config_path.name}: expected {key}={expected!r}, got {config.get(key)!r}"
                )
        if config["dataset"] != dataset:
            raise ValueError(f"{config_path.name}: dataset mismatch")

    if args.check_data:
        data_root = Path(catalog["data_root"])
        for dataset in TARGETS:
            graph = data_root / by_key[dataset]["relative_path"]
            if not graph.is_file():
                raise FileNotFoundError(graph)

    cells = [(source, target) for source in SOURCES for target in TARGETS]
    if len(cells) != 16 or len(set(cells)) != 16:
        raise AssertionError("pilot must contain exactly 16 unique transfer cells")
    print("social-specificity pilot: 4 sources x 4 targets = 16 cells")
    print("new training: cora seed 0, pubmed seed 0; 2,500 updates each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
