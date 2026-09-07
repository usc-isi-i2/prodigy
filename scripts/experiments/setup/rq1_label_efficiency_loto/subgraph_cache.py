"""Compact, shareable fixed-neighborhood cache for RQ1 adaptation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from torch_geometric.data import Data


def node_sampling_seed(target: str, seed: int, node: int) -> int:
    value = f"rq1-v2|{target}|{seed}|{node}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def build_compact_cache(dataset, nodes, *, target: str, seed: int, path: Path) -> None:
    entries = {}
    for position, node in enumerate(sorted({int(value) for value in nodes})):
        torch.manual_seed(node_sampling_seed(target, seed, node))
        graph = dataset[node]
        entries[node] = {
            "global_node_ids": graph.global_node_ids[:-1].clone(),
            "edge_index": graph.edge_index.clone(),
        }
        if (position + 1) % 500 == 0:
            print(f"cache {target} seed={seed}: {position + 1}/{len(nodes)}", flush=True)
    atomic_torch_save(
        {
            "format": "rq1-compact-neighborhood-v1",
            "target": target,
            "seed": seed,
            "entries": entries,
        },
        path,
    )


class CompactCachedSubgraphDataset:
    """Rehydrate cached topology with features from one resident full graph."""

    def __init__(self, dataset, cache_path: Path):
        self.dataset = dataset
        value = torch.load(cache_path, map_location="cpu", weights_only=False)
        if value.get("format") != "rq1-compact-neighborhood-v1":
            raise ValueError(f"unsupported neighborhood cache: {cache_path}")
        self.entries = value["entries"]
        self.cache = {}
        self.enabled = True
        self.hits = 0
        self.misses = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        index = int(index)
        if not self.enabled or index not in self.entries:
            return self.dataset[index]
        if index in self.cache:
            self.hits += 1
            return self.cache[index]
        entry = self.entries[index]
        node_ids = entry["global_node_ids"]
        x = self.dataset.graph.x[node_ids]
        x = torch.cat((x, torch.zeros(1, *x.shape[1:], dtype=x.dtype)), dim=0)
        supernode = len(node_ids)
        graph = Data(
            center_node_idx=index,
            global_node_ids=torch.cat((node_ids, torch.tensor([-1], dtype=torch.long))),
            edge_index=entry["edge_index"],
            num_nodes=supernode + 1,
            x=x,
            supernode=torch.tensor([supernode]),
            edge_index_supernode=torch.tensor([[0], [supernode]], dtype=torch.long),
            edge_index_from_supernode=torch.tensor([[supernode], [0]], dtype=torch.long),
        )
        self.cache[index] = graph
        self.misses += 1
        return graph
