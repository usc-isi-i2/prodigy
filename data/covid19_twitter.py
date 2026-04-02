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
    BinaryFutureLinkTask,
    _normalize_view_name,
    _load_named_tensor,
    _load_edge_feature_names,
    _apply_feature_subset,
    _apply_edge_feature_subset,
)


def _build_covid19_twitter_graph(raw: dict, **kwargs):
    edge_view = _normalize_view_name(kwargs.get("midterm_edge_view", "default"))
    edge_index, resolved_edge_view = _load_named_tensor(
        raw,
        edge_view,
        default_key="edge_index",
        views_key="edge_index_views",
        legacy_prefix="edge_index",
    )
    if edge_index is None:
        raise ValueError(
            f"No edge_index found for view '{resolved_edge_view}'. "
            "The graph file must contain at least 'edge_index'."
        )

    x = raw["x"]
    y = raw.get("y")
    if y is None:
        y = torch.full((x.shape[0],), -1, dtype=torch.long)
    graph = Data(x=x, edge_index=edge_index, y=y, num_nodes=x.shape[0])
    try:
        edge_attr, _ = _load_named_tensor(
            raw,
            resolved_edge_view,
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
    graph.handles = raw.get("handles", [])
    graph = _apply_feature_subset(graph, kwargs.get("midterm_feature_subset", "all"))
    graph = _apply_edge_feature_subset(graph, kwargs.get("midterm_edge_feature_subset", "all"), feature_names=edge_feature_names)
    return graph, resolved_edge_view


def get_covid19_twitter_dataset(root: str, n_hop: int = 1, graph_filename: str = "retweet_graph.pt", **kwargs) -> SubgraphDataset:
    graph_path = os.path.join(root, graph_filename)
    print(f"Loading covid19_twitter graph from {graph_path}...")
    raw = torch.load(graph_path, map_location="cpu")
    graph, resolved_edge_view = _build_covid19_twitter_graph(raw, **kwargs)
    print(f"Graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges, {graph.x.shape[1]} node features")
    print("Building neighbor sampler (CSR preprocessing)...", flush=True)
    neighbor_sampler = NeighborSampler(graph, num_hops=n_hop)
    print("Neighbor sampler ready.", flush=True)
    dataset = SubgraphDataset(graph, neighbor_sampler, bidirectional=False)
    if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
        print(f"Edge features: {graph.edge_attr.shape[1]} dims from edge view '{resolved_edge_view}'")

    task_name = kwargs.get("task_name", "")
    if task_name == "temporal_link_prediction":
        target_view = _normalize_view_name(kwargs.get("midterm_target_edge_view", "future"), default="future")
        future_edge_index, resolved_target_view = _load_named_tensor(
            raw,
            target_view,
            default_key="future_edge_index",
            views_key="target_edge_index_views",
            legacy_prefix="future_edge_index",
        )
        if future_edge_index is not None:
            print("Building future neighbor sampler...", flush=True)
            future_graph = Data(edge_index=future_edge_index, num_nodes=graph.num_nodes)
            dataset.future_neighbor_sampler = NeighborSampler(future_graph, num_hops=n_hop)
            dataset.future_edge_view = resolved_target_view
            print("Future neighbor sampler ready.", flush=True)
        else:
            dataset.future_edge_view = None
    else:
        dataset.future_edge_view = None
    return dataset


def get_covid19_twitter_dataloader(
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
    del root, bert, split_labels, train_cap, linear_probe, label_set, node_split
    seed = sum(ord(c) for c in split)
    graph = dataset.graph
    task_name = kwargs.get("task_name", "neighbor_matching")
    if task_name == "neighbor_matching":
        sampler = BatchSampler(
            batch_count,
            NeighborTask(dataset.neighbor_sampler, graph.num_nodes, "inout", kwargs.get("neighbor_sampling_strategy", "strict")),
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_embeddings = torch.zeros(1, 768).expand(graph.num_nodes, -1)
        is_multiway = True
    elif task_name == "temporal_link_prediction":
        if not hasattr(dataset, "future_neighbor_sampler"):
            raise ValueError("temporal_link_prediction requires target edges, but no future edge view was found.")
        neg_ratio = int(kwargs.get("midterm_lp_neg_ratio", 1))
        if isinstance(n_way, int):
            invalid_n_way = n_way != 1
        else:
            invalid_n_way = any(v != 1 for v in n_way)
        if invalid_n_way:
            raise ValueError(
                "temporal_link_prediction now only supports binary LP episodes. "
                f"Use --n_way 1, got n_way={n_way}."
            )
        task = BinaryFutureLinkTask(dataset.future_neighbor_sampler, graph.num_nodes, neg_ratio=neg_ratio)
        sampler = BatchSampler(
            batch_count,
            task,
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_embeddings = torch.zeros(1, 768).expand(graph.num_nodes, -1)
        is_multiway = False
    elif task_name == "classification":
        raise ValueError("covid19_twitter classification is unsupported until labels are added.")
    else:
        raise ValueError(f"Unknown task for covid19_twitter: {task_name}")

    aug_fn = get_aug(aug, graph.x) if (split == "train" or aug_test) else get_aug("")
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, collate_fn=Collator(label_embeddings, aug=aug_fn, is_multiway=is_multiway))
