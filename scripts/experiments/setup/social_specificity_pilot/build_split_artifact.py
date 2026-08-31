#!/usr/bin/env python3
"""Build a leakage-controlled 70/15/15 NM artifact for one citation graph."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.graph_construction.benchmark_targets import build_static_train_val_test_split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raw = torch.load(args.output, map_location="cpu", weights_only=False)
        edge_views = raw.get("edge_index_views", {})
        target_views = raw.get("target_edge_index_views", {})
        if "static_train" not in edge_views or not {
            "static_validation", "static_test"
        }.issubset(target_views):
            raise ValueError(f"existing output is not a complete three-way split: {args.output}")
        print(f"SKIP complete {args.output}")
        return 0
    raw = torch.load(args.input, map_location="cpu", weights_only=False)
    split = build_static_train_val_test_split(
        raw["edge_index"], validation_frac=0.15, test_frac=0.15, seed=0
    )
    raw.setdefault("edge_index_views", {})["static_train"] = split.train_edge_index
    raw.setdefault("target_edge_index_views", {})[
        "static_validation"
    ] = split.validation_edge_index
    raw.setdefault("target_edge_index_views", {})["static_test"] = split.test_edge_index
    raw.setdefault("benchmark_target_stats", {})[
        "static_train_validation_test"
    ] = split.stats
    raw["social_specificity_pilot_split_protocol"] = {
        "kind": "undirected_pair_70_15_15",
        "seed": 0,
        "input": str(args.input),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(raw, args.output)
    print(f"created {args.output}; stats={split.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
