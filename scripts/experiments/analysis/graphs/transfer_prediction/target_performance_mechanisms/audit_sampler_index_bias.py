#!/usr/bin/env python3
"""Reproduce an index-order dependence in the actual production member selector.

Uses a fixed random-walk return to isolate member selection from graph sampling.
Does not load data, train, or modify the production sampler.
"""
import argparse
import hashlib
import inspect
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
sys.path.insert(0, str(ROOT))

import torch
from data.dataloader import NeighborTask


class FixedWalk:
    def __init__(self, nodes):
        self.nodes = torch.tensor(nodes, dtype=torch.long)

    def random_walk(self, *args):
        return self.nodes.clone()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "data/sampler_index_audit.json")
    args = parser.parse_args()
    task = object.__new__(NeighborTask)
    task.direction = "inout"
    task.sampling_strategy = "strict"
    walk = list(range(10)) * 7
    task.neighbor_sampler = FixedWalk(walk)
    original = task._sample_center_members(99, 7, random.Random(0))
    # Relabel every endpoint consistently, preserving the underlying walk.
    permutation = {i: 9-i for i in range(10)}
    inverse = {v: k for k, v in permutation.items()}
    task.neighbor_sampler = FixedWalk([permutation[i] for i in walk])
    relabeled = task._sample_center_members(99, 7, random.Random(0))
    mapped_back = [inverse[i] for i in relabeled]
    result = {
        "selector_source_sha256": hashlib.sha256(inspect.getsource(NeighborTask._sample_center_members).encode()).hexdigest(),
        "torch_version": torch.__version__, "original_selected_nodes": original,
        "relabeled_selected_nodes_mapped_back": mapped_back,
        "selected_set_equal": set(original) == set(mapped_back),
        "original_support_nodes": original[:3], "original_query_nodes": original[3:],
        "interpretation": "Unique endpoints are sorted and the lowest seven are retained; the collator assigns the first three to support. Node renumbering changes selection and role. This reproduces a sampler property, not its impact on real graph performance.",
        "production_files": ["data/dataloader.py:NeighborTask._sample_center_members", "data/dataloader.py:Collator.process_one_task"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
