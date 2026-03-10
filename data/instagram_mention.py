import os
from typing import Optional, Set, Union

import torch
from torch.utils.data import DataLoader

from experiments.sampler import NeighborSampler
from .augment import get_aug
from .dataloader import NeighborTask, ParamSampler, BatchSampler, Collator
from .dataset import SubgraphDataset


def get_instagram_mention_dataset(
        root: str,
        graph_filename: str = "mention_graph.pt",
        n_hop: int = 2,
        **kwargs
) -> SubgraphDataset:
    graph_path = os.path.join(root, graph_filename)
    print(f"Loading Instagram mention graph from {graph_path}...")
    ckpt = torch.load(graph_path, map_location="cpu")
    graph = ckpt["data"]

    n_from_edges = int(graph.edge_index.max().item()) + 1 if graph.edge_index.numel() > 0 else 0
    n_from_x = graph.x.shape[0]
    num_nodes = max(n_from_edges, n_from_x)
    if num_nodes > n_from_x:
        pad = torch.zeros(num_nodes - n_from_x, graph.x.shape[1])
        graph.x = torch.cat([graph.x, pad], dim=0)
        print(f"Padded {num_nodes - n_from_x} feature-less nodes (edge-only) with zeros")
    graph.num_nodes = num_nodes

    print(f"Graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges, "
          f"{graph.x.shape[1]} features")

    neighbor_sampler = NeighborSampler(graph, num_hops=n_hop)
    return SubgraphDataset(graph, neighbor_sampler, bidirectional=False)


def get_instagram_mention_dataloader(
        dataset: SubgraphDataset,
        split: str,
        node_split: str,
        batch_size: Union[int, range],
        n_way: Union[int, range],
        n_shot: Union[int, range],
        n_query: Union[int, range],
        batch_count: int,
        num_workers: int,
        aug: str,
        aug_test: bool,
        **kwargs
) -> DataLoader:
    seed = sum(ord(c) for c in split)
    graph = dataset.graph

    sampler = BatchSampler(
        batch_count,
        NeighborTask(dataset.neighbor_sampler, graph.num_nodes, "inout"),
        ParamSampler(batch_size, n_way, n_shot, n_query, 1),
        seed=seed,
    )
    label_embeddings = torch.zeros(1, 768).expand(graph.num_nodes, -1)

    aug_fn = get_aug(aug, graph.x) if (split == "train" or aug_test) else get_aug("")

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=Collator(label_embeddings, aug=aug_fn),
    )
