#!/usr/bin/env python3
"""Extract frozen social-gfm pilot-v1 GraphSAGE node-history representations."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.eval.pair_link_ckpt import load_graph_blob

from .protocol import FeatureCache, save_feature_cache
from .targets import labeled_nodes, load_labels, selected_targets


EDGE_FAMILIES = 8
RESHARE_RELATION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def static_graph_arrays(edge_index: torch.Tensor, num_nodes: int, history_length: int):
    edge_index = edge_index.long().cpu()
    source, destination = edge_index
    out_degree = torch.bincount(source, minlength=num_nodes)
    in_degree = torch.bincount(destination, minlength=num_nodes)
    degree = torch.zeros((num_nodes, EDGE_FAMILIES * 2), dtype=torch.float32)
    degree[:, RESHARE_RELATION] = out_degree.float()
    degree[:, EDGE_FAMILIES + RESHARE_RELATION] = in_degree.float()
    degree.log1p_()
    degree /= degree.amax(dim=0, keepdim=True).clamp_min(1.0)

    centers = torch.cat((source, destination))
    directions = torch.cat((torch.zeros_like(source), torch.ones_like(destination)))
    order = torch.argsort(centers, stable=True)
    centers = centers[order]
    directions = directions[order]
    counts = torch.bincount(centers, minlength=num_nodes)
    starts = torch.cumsum(counts, dim=0) - counts
    positions = torch.arange(centers.numel()) - torch.repeat_interleave(starts, counts)
    keep = positions < history_length

    shape = (num_nodes, history_length)
    history_direction = torch.zeros(shape, dtype=torch.int8)
    history_mask = torch.zeros(shape, dtype=torch.bool)
    rows, cols = centers[keep], positions[keep]
    history_direction[rows, cols] = directions[keep].to(torch.int8)
    history_mask[rows, cols] = True
    history_relation = torch.full(shape, RESHARE_RELATION, dtype=torch.int8)
    return {
        "node_type": np.zeros(num_nodes, dtype=np.int8),
        "neighbor_type": np.zeros(shape, dtype=np.int8),
        "history_relation": history_relation.numpy(),
        "history_direction": history_direction.numpy(),
        "history_age": np.zeros(shape, dtype=np.float32),
        "history_time_observed": np.zeros(shape, dtype=np.float32),
        "history_mask": history_mask.numpy(),
        "degree": degree.numpy(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-id", default="graphsage_pilot_v1")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--targets", default="covid_political,election2020,ukr_rus_suspended,twibot20")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    sys.path.insert(0, str(args.repository.resolve()))
    from socialgfm.benchmark_models import LinkPredictor  # noqa: PLC0415
    from socialgfm.benchmark_run import _node_batch  # noqa: PLC0415

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("model") != "graphsage":
        raise ValueError(f"not the pilot GraphSAGE checkpoint: {args.checkpoint}")
    config = checkpoint["config"]
    device = torch.device(args.device)
    model = LinkPredictor(
        int(config["hidden_dim"]), int(config["history_length"]), False, False
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    checkpoint_hash = sha256_file(args.checkpoint)

    for target in selected_targets(args.targets):
        blob, graph = load_graph_blob(str(target.graph))
        labels = load_labels(blob, target.label_key)
        nodes = labeled_nodes(labels)
        edge_index = torch.as_tensor(graph.edge_index).long()
        arrays = static_graph_arrays(edge_index, int(graph.num_nodes), int(config["history_length"]))
        structural_graph = SimpleNamespace(
            record={"platform": "twitter", "key": target.name}, arrays=arrays
        )
        outputs = []
        with torch.no_grad():
            for start in range(0, nodes.size, args.batch_size):
                ids = nodes[start : start + args.batch_size]
                outputs.append(model.encoder(_node_batch(structural_graph, ids, device)).cpu())
        features = torch.cat(outputs).numpy()
        save_feature_cache(
            args.output_root / args.model_id / f"{target.name}.npz",
            FeatureCache(
                model_id=args.model_id,
                target=target.name,
                features=features,
                labels=labels,
                node_ids=nodes,
                metadata={
                    "architecture": "GraphSAGE",
                    "native_pretext": "pilot_v1_link_prediction",
                    "checkpoint": str(args.checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "training_seed": int(checkpoint.get("seed", config.get("seed", -1))),
                    "training_steps": int(config["train_steps"]),
                    "representation": "frozen_node_history_encoder",
                    "edge_view": "graph.edge_index",
                    "relation_mapping": "all target edges -> reshare",
                    "node_type_mapping": "all target nodes -> actor",
                    "history_length": int(config["history_length"]),
                    "label_key": target.label_key,
                },
            ),
        )
        print(f"{target.name}: labeled_nodes={nodes.size} dim={features.shape[1]}", flush=True)
        del blob, graph, arrays, structural_graph
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
