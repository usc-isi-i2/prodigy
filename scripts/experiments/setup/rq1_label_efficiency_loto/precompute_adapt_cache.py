#!/usr/bin/env python3
"""Precompute compact fixed neighborhoods once per target and adaptation seed."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.eval.pair_link_ckpt import load_graph_blob
from scripts.experiments.setup.adaptation_efficiency.extract_prodigy import classification_subgraph_dataset
from scripts.experiments.setup.adaptation_efficiency.protocol import sampled_labels, stratified_node_splits
from scripts.experiments.setup.adaptation_efficiency.targets import TARGETS, load_labels
from scripts.experiments.setup.rq1_label_efficiency_loto.adapt import fixed_eval_nodes
from scripts.experiments.setup.rq1_label_efficiency_loto.subgraph_cache import build_compact_cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(TARGETS), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.is_file():
        print(f"SKIP existing {args.output}")
        return 0
    target = TARGETS[args.target]
    blob, graph = load_graph_blob(str(target.graph))
    labels = load_labels(blob, target.label_key)
    splits = stratified_node_splits(labels, seed=0)
    nodes = []
    for budget in (1, 10, 100, 1000):
        nodes.extend(sampled_labels(labels, splits["train"], budget=budget, seed=args.seed))
    nodes.extend(fixed_eval_nodes(labels, splits["val"], 1000, seed=17000 + args.seed))
    nodes = np.unique(np.asarray(nodes, dtype=np.int64))
    torch.manual_seed(args.seed)
    dataset = classification_subgraph_dataset(graph, 2, [9, 9], 101)
    build_compact_cache(dataset, nodes, target=args.target, seed=args.seed, path=args.output)
    print(f"wrote {args.output} nodes={len(nodes)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
