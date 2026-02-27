import os
from typing import Optional, Set, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from experiments.sampler import NeighborSampler
from .augment import get_aug
from .dataloader import MulticlassTask, ParamSampler, BatchSampler, Collator, NeighborTask
from .dataset import SubgraphDataset


def get_midterm_dataset(
        root: str,
        n_hop: int = 2,
        **kwargs
) -> SubgraphDataset:
    graph_path = os.path.join(root, 'graph_data.pt')
    print(f"Loading midterm graph from {graph_path}...")
    raw = torch.load(graph_path, map_location='cpu')

    graph = Data(
        x=raw['x'],
        edge_index=raw['edge_index'],
        y=raw['y'],
        num_nodes=raw['x'].shape[0],
    )
    graph.label_names = raw['label_names']
    graph.feature_names = raw.get('feature_names', [])

    print(f"Graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges, "
          f"{len(graph.label_names)} state classes")
    labeled = (graph.y >= 0).sum().item()
    print(f"Labeled nodes: {labeled} / {graph.num_nodes} ({100 * labeled / graph.num_nodes:.1f}%)")

    print("Building neighbor sampler (CSR preprocessing)...", flush=True)
    neighbor_sampler = NeighborSampler(graph, num_hops=n_hop)
    print("Neighbor sampler ready.", flush=True)
    return SubgraphDataset(graph, neighbor_sampler, bidirectional=False)


def midterm_task(
        labels: np.ndarray,
        num_classes: int,
        split: str,
        label_set: Optional[Set[int]] = None,
        split_labels: bool = True,
        train_cap: Optional[int] = None,
        linear_probe: bool = False,
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
                disabled_idx = idx[train_cap:]
                train_label[disabled_idx] = -1 - i

    return MulticlassTask(labels, chosen_label_set, train_label, linear_probe)


def get_midterm_dataloader(
        dataset: SubgraphDataset,
        split: str,
        node_split: str,
        batch_size: Union[int, range],
        n_way: Union[int, range],
        n_shot: Union[int, range],
        n_query: Union[int, range],
        batch_count: int,
        root: str,
        bert,
        num_workers: int,
        aug: str,
        aug_test: bool,
        split_labels: bool,
        train_cap: Optional[int],
        linear_probe: bool,
        label_set: Optional[Set[int]] = None,
        **kwargs
) -> DataLoader:
    del node_split, root
    task_name = kwargs.get("task_name", "classification")
    seed = sum(ord(c) for c in split)

    graph = dataset.graph

    if task_name == "neighbor_matching":
        sampler = BatchSampler(
            batch_count,
            NeighborTask(dataset.neighbor_sampler, graph.num_nodes, "inout"),
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_embeddings = torch.zeros(1, 768).expand(graph.num_nodes, -1)
    elif task_name == "classification":
        label_names = list(getattr(graph, 'label_names', []))
        num_classes = len(label_names)

        if bert is not None:
            label_embeddings = bert.get_sentence_embeddings(label_names)
        else:
            label_embeddings = torch.randn(num_classes, 768)

        labels = graph.y.numpy()
        task = midterm_task(
            labels=labels,
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
        raise ValueError(f"Unknown task for midterm: {task_name}")

    aug_fn = get_aug(aug, dataset.graph.x) if (split == "train" or aug_test) else get_aug("")

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=Collator(label_embeddings, aug=aug_fn),
    )
