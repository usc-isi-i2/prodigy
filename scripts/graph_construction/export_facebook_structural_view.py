#!/usr/bin/env python3
"""Export the edge-participating Facebook pages as a compact runtime graph."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch_geometric.data import Data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def remap_edges(edge_index: torch.Tensor, old_to_new: torch.Tensor) -> torch.Tensor:
    remapped = old_to_new[edge_index]
    if bool((remapped < 0).any()):
        raise ValueError("An edge view contains a node outside structural_node_mask")
    return remapped


def subset_node_tensor(value, mask: torch.Tensor):
    if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == mask.numel():
        return value[mask]
    return value


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.out.expanduser().resolve()
    meta_path = output_path.with_suffix(".meta.json")
    if output_path.exists() or meta_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path} or {meta_path}")

    raw = torch.load(input_path, map_location="cpu")
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict graph artifact, got {type(raw).__name__}")
    mask = raw.get("node_attributes", {}).get("structural_node_mask")
    if mask is None or mask.dtype != torch.bool or mask.ndim != 1:
        raise ValueError("Missing one-dimensional boolean structural_node_mask")
    structural_indices = torch.where(mask)[0]
    old_to_new = torch.full((mask.numel(),), -1, dtype=torch.long)
    old_to_new[structural_indices] = torch.arange(structural_indices.numel())

    graph = dict(raw)
    graph["x"] = raw["x"][mask]
    graph["y"] = raw["y"][mask]
    graph["edge_index"] = remap_edges(raw["edge_index"], old_to_new)
    graph["edge_index_views"] = {
        name: remap_edges(value, old_to_new)
        for name, value in raw.get("edge_index_views", {}).items()
    }
    graph["target_edge_index_views"] = {
        name: remap_edges(value, old_to_new)
        for name, value in raw.get("target_edge_index_views", {}).items()
    }
    if raw.get("future_edge_index") is not None:
        graph["future_edge_index"] = remap_edges(raw["future_edge_index"], old_to_new)
    graph["node_targets"] = {
        name: subset_node_tensor(value, mask)
        for name, value in raw.get("node_targets", {}).items()
    }
    graph["node_classification_targets"] = {
        name: subset_node_tensor(value, mask)
        for name, value in raw.get("node_classification_targets", {}).items()
    }
    graph["node_split_masks"] = {
        name: subset_node_tensor(value, mask)
        for name, value in raw.get("node_split_masks", {}).items()
    }
    graph["node_attributes"] = {
        name: subset_node_tensor(value, mask)
        for name, value in raw.get("node_attributes", {}).items()
    }
    graph["node_attributes"]["structural_node_mask"] = torch.ones(
        structural_indices.numel(), dtype=torch.bool
    )
    user_ids = raw.get("user_ids", [])
    if len(user_ids) == mask.numel():
        graph["user_ids"] = [user_ids[index] for index in structural_indices.tolist()]
        graph["u2i"] = {user_id: index for index, user_id in enumerate(graph["user_ids"])}
    else:
        graph.pop("u2i", None)

    data = Data(
        x=graph["x"],
        edge_index=graph["edge_index"],
        edge_attr=graph.get("edge_attr"),
        y=graph["y"],
        num_nodes=structural_indices.numel(),
    )
    data.feature_names = graph.get("feature_names", [])
    data.edge_attr_feature_names = graph.get("edge_attr_feature_names", [])
    data.label_names = graph.get("label_names", [])
    data.user_ids = graph.get("user_ids", [])
    data.structural_node_mask = graph["node_attributes"]["structural_node_mask"]
    graph["data"] = data

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")
    torch.save(graph, temporary)
    temporary.rename(output_path)
    metadata = {
        "source_artifact": str(input_path),
        "nodes": int(graph["x"].shape[0]),
        "edges": int(graph["edge_index"].shape[1]),
        "node_feature_dim": int(graph["x"].shape[1]),
        "labels": len(graph.get("label_names", [])),
        "policy": "induced runtime view over structural_node_mask; canonical artifact unchanged",
    }
    temporary_meta = meta_path.with_name(meta_path.name + f".tmp.{os.getpid()}")
    temporary_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    temporary_meta.rename(meta_path)
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
