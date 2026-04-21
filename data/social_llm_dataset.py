"""
Generic dataset/dataloader for social_llm_* graph datasets.

Supports: covid_mf, election2020, hate_bots05, hate_bots08, ukr_rus_hate, ukr_rus_suspended.

Each dataset's graph .pt is built by data/data/social_llm/scripts/generate_graph.py.
For multi-label datasets (covid_mf, hate_bots05/08, ukr_rus_hate), re-run
generate_graph.py with --label_col <col> --out retweet_graph_<col>.pt to get
a graph for a specific label, then use --graph_filename to select it at train time.
"""
import os
from typing import Optional, Set, Union

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from experiments.sampler import NeighborSampler
from .augment import get_aug
from .dataloader import ParamSampler, BatchSampler, Collator, NeighborTask
from .dataset import SubgraphDataset
from .midterm import (
    _normalize_view_name,
    _load_named_tensor,
    _load_edge_feature_names,
    _apply_feature_subset,
    _apply_edge_feature_subset,
    _build_stratified_node_splits,
    _mask_labels_to_node_split,
    _apply_label_downsample,
    midterm_task,
    _deterministic_label_embeddings,
)


def _build_graph(raw: dict, **kwargs):
    edge_view = _normalize_view_name(kwargs.get("midterm_edge_view", "default"))
    edge_index, resolved_edge_view = _load_named_tensor(
        raw, edge_view,
        default_key="edge_index",
        views_key="edge_index_views",
        legacy_prefix="edge_index",
    )
    if edge_index is None:
        raise ValueError(f"No edge_index found for view '{resolved_edge_view}'.")

    x = raw["x"]
    y = raw.get("y")
    if y is None:
        y = torch.full((x.shape[0],), -1, dtype=torch.long)

    graph = Data(x=x, edge_index=edge_index, y=y, num_nodes=x.shape[0])

    try:
        edge_attr, _ = _load_named_tensor(
            raw, resolved_edge_view,
            default_key="edge_attr",
            views_key="edge_attr_views",
            legacy_prefix="edge_attr",
        )
    except ValueError:
        edge_attr = None

    edge_feature_names = _load_edge_feature_names(raw, resolved_edge_view)
    if edge_attr is not None:
        graph.edge_attr = edge_attr
    graph.edge_attr_feature_names = edge_feature_names
    graph.feature_names = raw.get("feature_names", [])
    graph.label_names = raw.get("label_names", [])
    graph.user_ids = raw.get("user_ids", [])
    graph.y = _apply_label_downsample(
        graph.y, graph.label_names,
        kwargs.get("midterm_label_downsample", ""),
        seed=int(kwargs.get("seed", 0) or 0),
    )
    graph = _apply_feature_subset(graph, kwargs.get("midterm_feature_subset", "all"))
    graph = _apply_edge_feature_subset(
        graph, kwargs.get("midterm_edge_feature_subset", "all"),
        feature_names=edge_feature_names,
    )
    return graph, resolved_edge_view


def _get_dataset(dataset_name: str, root: str, n_hop: int = 1,
                 graph_filename: str = "retweet_graph.pt", **kwargs) -> SubgraphDataset:
    graph_path = os.path.join(root, graph_filename)
    print(f"Loading {dataset_name} graph from {graph_path}...")
    raw = torch.load(graph_path, map_location="cpu")
    graph, resolved_edge_view = _build_graph(raw, **kwargs)
    print(f"Graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges, "
          f"{graph.x.shape[1]} node features  labels={graph.label_names}")
    print("Building neighbor sampler...", flush=True)
    neighbor_sampler = NeighborSampler(graph, num_hops=n_hop)
    print("Neighbor sampler ready.", flush=True)
    dataset = SubgraphDataset(graph, neighbor_sampler, bidirectional=False)
    if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
        print(f"Edge features: {graph.edge_attr.shape[1]} dims from '{resolved_edge_view}'")
    dataset.future_edge_view = None
    return dataset


def _get_dataloader(dataset_name: str, dataset: SubgraphDataset, split: str,
                    node_split: str, batch_size, n_way, n_shot, n_query,
                    batch_count: int, root: str, bert, num_workers: int,
                    aug: str, aug_test: bool, split_labels: bool,
                    train_cap: Optional[int], linear_probe: bool,
                    label_set: Optional[Set[int]] = None, **kwargs) -> DataLoader:
    del root
    seed = sum(ord(c) for c in split)
    graph = dataset.graph
    task_name = kwargs.get("task_name", "neighbor_matching")

    if task_name == "neighbor_matching":
        sampler = BatchSampler(
            batch_count,
            NeighborTask(
                dataset.neighbor_sampler, graph.num_nodes, "inout",
                kwargs.get("neighbor_sampling_strategy", "strict"),
            ),
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_embeddings = torch.zeros(1, 768).expand(graph.num_nodes, -1)
        is_multiway = True

    elif task_name == "classification":
        label_names = list(getattr(graph, "label_names", []))
        num_classes = len(label_names)
        if num_classes == 0:
            raise ValueError(f"{dataset_name} classification requires graph.label_names.")

        label_embeddings = (
            bert.get_sentence_embeddings(label_names)
            if bert is not None
            else _deterministic_label_embeddings(label_names, dim=768)
        )

        labels = graph.y.numpy()
        if not hasattr(dataset, "_classification_node_splits"):
            dataset._classification_node_splits = _build_stratified_node_splits(labels, seed=0)

        split_key = (node_split or "").strip() or split
        node_splits = dataset._classification_node_splits
        if split_key not in node_splits:
            raise ValueError(
                f"Unknown {dataset_name} split '{split_key}'. "
                f"Available: {sorted(node_splits.keys())}"
            )
        labels = _mask_labels_to_node_split(labels, node_splits[split_key])
        task = midterm_task(
            labels=labels, num_classes=num_classes, split=split,
            label_set=label_set, split_labels=False,
            train_cap=train_cap, linear_probe=linear_probe,
        )
        task.original_graph_labels = graph.y.numpy().copy()
        task.split_masked_labels = labels.copy()
        sampler = BatchSampler(
            batch_count, task,
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        is_multiway = True

    else:
        raise ValueError(f"Unknown task for {dataset_name}: {task_name}")

    aug_fn = get_aug(aug, graph.x) if (split == "train" or aug_test) else get_aug("")
    return DataLoader(
        dataset, batch_sampler=sampler, num_workers=num_workers,
        collate_fn=Collator(label_embeddings, aug=aug_fn, is_multiway=is_multiway),
    )


# ---------------------------------------------------------------------------
# Per-dataset public API  (trainer.py / data_loader_wrapper.py call these)
# ---------------------------------------------------------------------------

def get_covid_mf_dataset(root, **kwargs):
    return _get_dataset("covid_mf", root, **kwargs)

def get_covid_mf_dataloader(*args, **kwargs):
    return _get_dataloader("covid_mf", *args, **kwargs)


def get_election2020_dataset(root, **kwargs):
    return _get_dataset("election2020", root, **kwargs)

def get_election2020_dataloader(*args, **kwargs):
    return _get_dataloader("election2020", *args, **kwargs)


def get_hate_bots05_dataset(root, **kwargs):
    return _get_dataset("hate_bots05", root, **kwargs)

def get_hate_bots05_dataloader(*args, **kwargs):
    return _get_dataloader("hate_bots05", *args, **kwargs)


def get_hate_bots08_dataset(root, **kwargs):
    return _get_dataset("hate_bots08", root, **kwargs)

def get_hate_bots08_dataloader(*args, **kwargs):
    return _get_dataloader("hate_bots08", *args, **kwargs)


def get_ukr_rus_hate_dataset(root, **kwargs):
    return _get_dataset("ukr_rus_hate", root, **kwargs)

def get_ukr_rus_hate_dataloader(*args, **kwargs):
    return _get_dataloader("ukr_rus_hate", *args, **kwargs)


def get_ukr_rus_suspended_dataset(root, **kwargs):
    return _get_dataset("ukr_rus_suspended", root, **kwargs)

def get_ukr_rus_suspended_dataloader(*args, **kwargs):
    return _get_dataloader("ukr_rus_suspended", *args, **kwargs)
