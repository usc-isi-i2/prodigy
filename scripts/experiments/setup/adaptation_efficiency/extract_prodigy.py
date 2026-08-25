#!/usr/bin/env python3
"""Extract PRODIGY pooled subgraph embeddings before metagraph label propagation."""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
import torch

from scripts.eval.pair_link_ckpt import (
    ENCODER_DEFAULTS,
    build_subgraph_dataset,
    load_frozen_encoder,
    load_graph_blob,
)
from scripts.eval.pair_link_eval import embeddings_by_node

from .protocol import FeatureCache, save_feature_cache
from .targets import labeled_nodes, load_labels, selected_targets


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--targets", default="covid_political,election2020,ukr_rus_suspended,twibot20")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--background-view", default="static_train")
    parser.add_argument("--n-hop", type=int, default=2)
    parser.add_argument("--hop-sizes", default="9,9")
    parser.add_argument("--node-limit", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    hops = [int(value) for value in args.hop_sizes.split(",") if value]
    if len(hops) != args.n_hop:
        raise ValueError("hop count mismatch")
    params = dict(ENCODER_DEFAULTS)
    model = load_frozen_encoder(args.checkpoint, params, device=args.device)
    checkpoint_hash = sha256_file(args.checkpoint)

    for target in selected_targets(args.targets):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        blob, graph = load_graph_blob(str(target.graph))
        labels = load_labels(blob, target.label_key)
        nodes = labeled_nodes(labels)
        dataset = build_subgraph_dataset(
            blob,
            graph,
            args.n_hop,
            args.background_view,
            hop_sizes=hops,
            node_limit=args.node_limit,
        )
        embedded = embeddings_by_node(
            model,
            dataset,
            nodes,
            int(graph.num_nodes),
            device=args.device,
            batch_size=args.batch_size,
        )
        features = embedded.table
        save_feature_cache(
            args.output_root / args.model_id / f"{target.name}.npz",
            FeatureCache(
                model_id=args.model_id,
                target=target.name,
                features=features,
                labels=labels,
                node_ids=nodes,
                metadata={
                    "architecture": "PRODIGY",
                    "native_pretext": "neighbor_matching",
                    "checkpoint": str(args.checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "checkpoint_step": 2500,
                    "training_seed": args.training_seed,
                    "representation": "pooled_subgraph_embedding_before_metagraph",
                    "background_view": args.background_view,
                    "n_hop": args.n_hop,
                    "hop_sizes": hops,
                    "node_limit": args.node_limit,
                    "sampling_seed": 0,
                    "label_key": target.label_key,
                },
            ),
        )
        print(f"{target.name}: labeled_nodes={nodes.size} dim={features.shape[1]}", flush=True)
        del blob, graph, dataset, embedded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
