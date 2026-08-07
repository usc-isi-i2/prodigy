#!/usr/bin/env python3
"""Create the immutable 70/15/15 all-nine artifact for the final core run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.graph_construction.benchmark_targets import build_static_train_val_test_split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    print(f"loading {args.input}", flush=True)
    started = time.time()
    raw = torch.load(args.input, map_location="cpu")
    print(f"loaded in {time.time() - started:.1f}s", flush=True)
    split = build_static_train_val_test_split(
        raw["edge_index"], validation_frac=0.15, test_frac=0.15, seed=0
    )
    stats = split.stats
    raw.setdefault("edge_index_views", {})["static_train"] = split.train_edge_index
    raw.setdefault("target_edge_index_views", {})["static_validation"] = split.validation_edge_index
    raw.setdefault("target_edge_index_views", {})["static_test"] = split.test_edge_index
    raw.setdefault("benchmark_target_stats", {})["static_train_validation_test"] = stats
    raw["final_core_split_protocol"] = {
        "kind": "undirected_pair_70_15_15",
        "seed": 0,
        "input": str(args.input),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    del split
    print(f"saving {args.output}; stats={stats}", flush=True)
    torch.save(raw, args.output)
    print(f"complete in {time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
