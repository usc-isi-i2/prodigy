import os
from typing import Optional, Set, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.sampler import NeighborSampler
from .augment import get_aug
from .dataloader import MulticlassTask, NeighborTask, ParamSampler, BatchSampler, Collator
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
        pad_n = num_nodes - n_from_x
        graph.x = torch.cat([graph.x, torch.zeros(pad_n, graph.x.shape[1])], dim=0)
        print(f"Padded {pad_n} feature-less nodes with zeros")

    # Pad y if present (edge-only nodes get label -1)
    if hasattr(graph, "y") and graph.y is not None:
        if graph.y.shape[0] < num_nodes:
            pad_n = num_nodes - graph.y.shape[0]
            graph.y = torch.cat([graph.y, torch.full((pad_n,), -1, dtype=torch.long)])

    graph.num_nodes = num_nodes
    print(f"Graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges, "
          f"{graph.x.shape[1]} features")

    neighbor_sampler = NeighborSampler(graph, num_hops=n_hop)
    return SubgraphDataset(graph, neighbor_sampler, bidirectional=False)


def _classification_task(
        labels: np.ndarray,
        num_classes: int,
        split: str,
        label_set: Optional[Set[int]],
        split_labels: bool,
        train_cap: Optional[int],
        linear_probe: bool,
) -> MulticlassTask:
    if label_set is not None:
        chosen_label_set = set(label_set)
    elif split_labels:
        all_labels = list(range(num_classes))
        n_train = max(1, int(num_classes * 0.6))
        n_val = max(1, int(num_classes * 0.2))
        train_labels = all_labels[:n_train]
        val_labels = all_labels[n_train:n_train + n_val]
        test_labels = all_labels[n_train + n_val:]
        if not test_labels and val_labels:
            test_labels = [val_labels.pop()]
        if not val_labels and train_labels:
            val_labels = [train_labels.pop()]
        if split == "train":
            chosen_label_set = set(train_labels)
        elif split == "val":
            chosen_label_set = set(val_labels)
        elif split == "test":
            chosen_label_set = set(test_labels)
        else:
            raise ValueError(f"Invalid split: {split}")
    else:
        chosen_label_set = set(range(num_classes))

    train_label = None
    if train_cap is not None and split == "train":
        train_label = labels.copy()
        for i in range(num_classes):
            idx = np.where(labels == i)[0]
            if len(idx) > train_cap:
                train_label[idx[train_cap:]] = -1 - i

    return MulticlassTask(labels, chosen_label_set, train_label, linear_probe)


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
        bert=None,
        split_labels: bool = True,
        train_cap: Optional[int] = None,
        linear_probe: bool = False,
        label_set: Optional[Set[int]] = None,
        **kwargs
) -> DataLoader:
    seed = sum(ord(c) for c in split)
    graph = dataset.graph
    task_name = kwargs.get("task_name", "neighbor_matching")

    if task_name == "neighbor_matching":
        sampler = BatchSampler(
            batch_count,
            NeighborTask(
                dataset.neighbor_sampler,
                graph.num_nodes,
                "inout",
                kwargs.get("neighbor_sampling_strategy", "strict"),
            ),
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_embeddings = torch.zeros(1, 768).expand(graph.num_nodes, -1)

    elif task_name == "classification":
        label_names = list(getattr(graph, "label_names", []))
        num_classes = len(label_names)

        if bert is not None:
            label_embeddings = bert.get_sentence_embeddings(label_names)
        else:
            label_embeddings = torch.randn(num_classes, 768)

        task = _classification_task(
            labels=graph.y.numpy(),
            num_classes=num_classes,
            split=split,
            label_set=label_set,
            split_labels=split_labels,
            train_cap=train_cap,
            linear_probe=linear_probe,
        )
        sampler = BatchSampler(
            batch_count,
            task,
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown task for instagram_mention: {task_name}")

    aug_fn = get_aug(aug, graph.x) if (split == "train" or aug_test) else get_aug("")

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=Collator(label_embeddings, aug=aug_fn),
    )
