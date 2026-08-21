from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from .data import GraphArtifact, _undirected_pairs


PARTITIONS = ("train", "validation", "test")


@dataclass(frozen=True)
class NodeSplit:
    train: torch.Tensor
    validation: torch.Tensor
    test: torch.Tensor
    seed: int
    fractions: tuple[float, float, float]

    def indices(self, partition: str) -> torch.Tensor:
        if partition not in PARTITIONS:
            raise ValueError(f"unknown partition: {partition}")
        return getattr(self, partition)


def _allocate(indices: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = indices[torch.randperm(len(indices), generator=generator)]
    size = len(indices)
    if size < 3:
        raise ValueError(f"cannot create three nonempty partitions from {size} nodes")
    n_train = max(1, int(size * 0.70))
    n_validation = max(1, int(size * 0.15))
    if n_train + n_validation >= size:
        n_train = size - 2
        n_validation = 1
    return (
        indices[:n_train],
        indices[n_train : n_train + n_validation],
        indices[n_train + n_validation :],
    )


def stratified_node_split(y: torch.Tensor | None, num_nodes: int, seed: int) -> NodeSplit:
    generator = torch.Generator().manual_seed(seed)
    buckets: list[torch.Tensor] = []
    if y is not None:
        labels = y.reshape(-1).long()
        labeled = torch.isfinite(labels.float()) & (labels >= 0)
        for label in torch.unique(labels[labeled], sorted=True):
            buckets.append(torch.nonzero(labeled & (labels == label), as_tuple=False).reshape(-1))
        unlabeled = torch.nonzero(~labeled, as_tuple=False).reshape(-1)
        if len(unlabeled):
            buckets.append(unlabeled)
    else:
        buckets.append(torch.arange(num_nodes))
    allocated = [_allocate(bucket, generator) for bucket in buckets]
    parts = []
    for position in range(3):
        values = torch.cat([item[position] for item in allocated]).long()
        parts.append(values[torch.randperm(len(values), generator=generator)].contiguous())
    split = NodeSplit(parts[0], parts[1], parts[2], seed, (0.70, 0.15, 0.15))
    validate_node_split(split, num_nodes)
    return split


def validate_node_split(split: NodeSplit, num_nodes: int) -> None:
    sets = [set(split.indices(name).tolist()) for name in PARTITIONS]
    if any(not values for values in sets):
        raise ValueError("all node partitions must be nonempty")
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("node partitions overlap")
    if sets[0] | sets[1] | sets[2] != set(range(num_nodes)):
        raise ValueError("node partitions do not cover the graph")


def split_hash(split: NodeSplit) -> str:
    digest = hashlib.sha256()
    for partition in PARTITIONS:
        digest.update(partition.encode())
        digest.update(split.indices(partition).numpy().astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def save_split(path: Path, split: NodeSplit, graph_name: str, num_nodes: int) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite split: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "graph": graph_name,
            "num_nodes": num_nodes,
            "seed": split.seed,
            "fractions": split.fractions,
            "train": split.train,
            "validation": split.validation,
            "test": split.test,
            "sha256": split_hash(split),
        },
        path,
    )


def load_split(path: Path, graph_name: str, num_nodes: int) -> NodeSplit:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if raw["graph"] != graph_name or int(raw["num_nodes"]) != num_nodes:
        raise ValueError(f"split metadata mismatch: {path}")
    split = NodeSplit(
        raw["train"].long(), raw["validation"].long(), raw["test"].long(),
        int(raw["seed"]), tuple(raw["fractions"]),
    )
    validate_node_split(split, num_nodes)
    if raw.get("sha256") != split_hash(split):
        raise ValueError(f"split checksum mismatch: {path}")
    return split


def load_raw(path: str | Path) -> dict:
    raw = torch.load(Path(path), map_location="cpu", weights_only=False)
    return raw if isinstance(raw, dict) else raw.to_dict()


def induced_partition(
    name: str,
    path: str | Path,
    split: NodeSplit,
    partition: str,
) -> GraphArtifact:
    raw = load_raw(path)
    x, edge_index, y = raw.get("x"), raw.get("edge_index"), raw.get("y")
    nodes = split.indices(partition)
    induced_edges, _ = subgraph(nodes, edge_index.long(), relabel_nodes=True, num_nodes=x.shape[0])
    pairs = _undirected_pairs(induced_edges).T.contiguous()
    if not pairs.numel():
        raise ValueError(f"{name}/{partition}: induced partition has no edges")
    background = torch.cat((pairs, pairs.flip(0)), dim=1)
    data = Data(
        x=x[nodes].float(),
        edge_index=background,
        y=None if y is None else y.reshape(-1)[nodes].long(),
        original_node_id=nodes,
    )
    return GraphArtifact(f"{name}:{partition}", Path(path), data, pairs, pairs)


def split_summary(graph_name: str, split: NodeSplit, y: torch.Tensor | None) -> dict:
    result = {"graph": graph_name, "seed": split.seed, "sha256": split_hash(split), "partitions": {}}
    for partition in PARTITIONS:
        indices = split.indices(partition)
        counts = {}
        if y is not None:
            labels = y.reshape(-1)[indices]
            counts = {str(int(label)): int((labels == label).sum()) for label in torch.unique(labels[labels >= 0], sorted=True)}
        result["partitions"][partition] = {"nodes": len(indices), "class_counts": counts}
    return result
