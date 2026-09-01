import os
from typing import Optional, Set, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from experiments.sampler import NeighborSampler, sampler_kwargs_from_config
from .augment import get_aug
from .dataloader import ParamSampler, BatchSampler, Collator, NeighborTask, RegressionTask, MultiTaskSplitBatch
from .dataset import SubgraphDataset
from .neighbor_matching_split import (
    configure_edge_split,
    ensure_static_views,
    positive_sampler_for_split,
)
from .midterm import (
    BinaryFutureLinkTask,
    StaticLinkTask,
    _normalize_view_name,
    _load_named_tensor,
    _load_edge_feature_names,
    _select_target_from_feature,
    _apply_feature_subset,
    _apply_edge_feature_subset,
    _build_stratified_node_splits,
    _build_regression_node_splits,
    _mask_labels_to_node_split,
    _apply_label_downsample,
    midterm_task,
    _deterministic_label_embeddings,
    _label_embedding_texts,
)


def _build_ukr_rus_twitter_graph(raw: dict, **kwargs):
    edge_view = _normalize_view_name(kwargs.get("edge_view", kwargs.get("midterm_edge_view", "default")))
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

    graph = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=x.shape[0],
    )
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
    graph.label_type = raw.get("label_type", "classification")
    graph.user_ids = raw.get("user_ids", [])
    graph.node_targets = raw.get("node_targets", None)
    graph = _select_target_from_feature(
        graph,
        kwargs.get("target_feature", ""),
        keep_in_x=bool(kwargs.get("target_feature_keep_in_x", False)),
        transform=kwargs.get("target_transform", "none"),
    )
    if graph.label_type != "regression":
        graph.y = _apply_label_downsample(
            graph.y,
            graph.label_names,
            kwargs.get("midterm_label_downsample", ""),
            seed=int(kwargs.get("seed", 0) or 0),
        )

    graph = _apply_feature_subset(graph, kwargs.get("feature_subset", kwargs.get("midterm_feature_subset", "all")))
    graph = _apply_edge_feature_subset(
        graph,
        kwargs.get("edge_feature_subset", kwargs.get("midterm_edge_feature_subset", "all")),
        feature_names=edge_feature_names,
    )
    return graph, resolved_edge_view


def get_ukr_rus_twitter_dataset(
        root: str,
        n_hop: int = 1,
        graph_filename: str = "retweet_graph.pt",
        **kwargs
) -> SubgraphDataset:
    graph_path = os.path.join(root, graph_filename)
    print(f"Loading ukr_rus_twitter graph from {graph_path}...")
    raw = torch.load(graph_path, map_location="cpu")
    ensure_static_views(raw, kwargs)

    task_name = kwargs.get("task_name", "")
    if task_name == "static_link_prediction" and not kwargs.get("edge_view") and not kwargs.get("midterm_edge_view"):
        kwargs = {**kwargs, "edge_view": "static_background"}
        print("static_link_prediction: defaulting to 'static_background' edge view for subgraph construction.")

    graph, resolved_edge_view = _build_ukr_rus_twitter_graph(raw, **kwargs)

    print(
        f"Graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges, "
        f"{graph.x.shape[1]} node features"
    )
    print("Building neighbor sampler (CSR preprocessing)...", flush=True)
    sampler_kwargs = sampler_kwargs_from_config(kwargs, n_hop)
    neighbor_sampler = NeighborSampler(graph, num_hops=n_hop, **sampler_kwargs)
    print("Neighbor sampler ready.", flush=True)
    dataset = SubgraphDataset(graph, neighbor_sampler, bidirectional=False)
    if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
        print(f"Edge features: {graph.edge_attr.shape[1]} dims from edge view '{resolved_edge_view}'")

    if task_name in ("temporal_link_prediction", "static_link_prediction"):
        default_target = "static_holdout" if task_name == "static_link_prediction" else "future"
        target_view = _normalize_view_name(
            kwargs.get("target_edge_view", kwargs.get("midterm_target_edge_view", default_target)),
            default=default_target,
        )
        future_edge_index, resolved_target_view = _load_named_tensor(
            raw,
            target_view,
            default_key="future_edge_index",
            views_key="target_edge_index_views",
            legacy_prefix="future_edge_index",
        )
        if future_edge_index is not None:
            print("Building target-edge neighbor sampler...", flush=True)
            future_graph = Data(edge_index=future_edge_index, num_nodes=graph.num_nodes)
            dataset.future_neighbor_sampler = NeighborSampler(
                future_graph, num_hops=n_hop, **sampler_kwargs
            )
            dataset.future_edge_view = resolved_target_view
            print("Target-edge neighbor sampler ready.", flush=True)
        else:
            dataset.future_edge_view = None
    else:
        dataset.future_edge_view = None
    configure_edge_split(
        raw, dataset, kwargs, n_hop=n_hop, sampler_kwargs=sampler_kwargs
    )
    return dataset


def get_ukr_rus_twitter_dataloader(
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
    del root
    seed = sum(ord(c) for c in split) + int(
        kwargs.get("eval_episode_seed_offset", 0) or 0
    )
    graph = dataset.graph
    task_name = kwargs.get("task_name", "neighbor_matching")

    if task_name == "neighbor_matching":
        positive_sampler = positive_sampler_for_split(dataset, split, kwargs)
        sampler = BatchSampler(
            batch_count,
            NeighborTask(
                positive_sampler,
                graph.num_nodes,
                "inout",
                kwargs.get("neighbor_sampling_strategy", "strict"),
                filter_min_degree=bool(kwargs.get("neighbor_matching_edge_split", False)),
            ),
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_embeddings = torch.zeros(1, 768).expand(graph.num_nodes, -1)
        is_multiway = True
    elif task_name == "temporal_link_prediction":
        if not hasattr(dataset, "future_neighbor_sampler"):
            raise ValueError(
                "temporal_link_prediction requires target edges, but no future edge view was found. "
                "Provide --target_edge_view (or ensure 'future_edge_index' exists in graph_data.pt)."
            )
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
        print(
            "Using binary temporal LP sampler "
            f"(explicit future-positive vs non-future-negative pairs, neg_ratio={neg_ratio}:1)."
        )
        task = BinaryFutureLinkTask(
            dataset.future_neighbor_sampler,
            graph.num_nodes,
            neg_ratio=neg_ratio,
        )
        sampler = BatchSampler(
            batch_count,
            task,
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_embeddings = torch.zeros(1, 768).expand(graph.num_nodes, -1)
        is_multiway = False
    elif task_name == "static_link_prediction":
        if not hasattr(dataset, "future_neighbor_sampler"):
            raise ValueError(
                "static_link_prediction requires a 'static_holdout' target edge view; "
                "enrich the graph with benchmark targets (enrich_graph_targets.py)."
            )
        neg_ratio = int(kwargs.get("midterm_lp_neg_ratio", 1))
        hard_negatives = bool(kwargs.get("hard_negatives", True))
        invalid_n_way = (n_way != 1) if isinstance(n_way, int) else any(v != 1 for v in n_way)
        if invalid_n_way:
            raise ValueError(f"static_link_prediction only supports n_way=1, got n_way={n_way}.")
        print(
            f"Using static LP sampler (held-out edges vs "
            f"{'hard 2-hop' if hard_negatives else 'random'} negatives, neg_ratio={neg_ratio}:1)."
        )
        task = StaticLinkTask(
            dataset.future_neighbor_sampler,
            dataset.neighbor_sampler,
            graph.num_nodes,
            neg_ratio=neg_ratio,
            hard_negatives=hard_negatives,
        )
        sampler = BatchSampler(
            batch_count,
            task,
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        label_embeddings = torch.zeros(1, 768).expand(graph.num_nodes, -1)
        is_multiway = False
    elif task_name in {"classification", "cls_nm"}:
        if getattr(graph, "label_type", "classification") == "regression":
            raise ValueError(
                "ukr_rus_twitter graph stores regression labels; use --task_name regression, not classification."
            )
        label_names = list(getattr(graph, "label_names", []))
        num_classes = len(label_names)
        if num_classes == 0:
            raise ValueError("ukr_rus_twitter classification requires graph.label_names to be populated.")

        if bert is not None:
            label_texts = _label_embedding_texts(label_names, kwargs.get("label_emb_texts", ""))
            label_embeddings = bert.get_sentence_embeddings(label_texts)
        else:
            label_embeddings = _deterministic_label_embeddings(label_names, dim=768)

        labels = graph.y.numpy()
        if not hasattr(dataset, "_classification_node_splits"):
            dataset._classification_node_splits = _build_stratified_node_splits(labels, seed=0)

        split_key = (node_split or "").strip() or split
        node_splits = dataset._classification_node_splits
        if split_key not in node_splits:
            raise ValueError(
                f"Unknown ukr_rus_twitter classification node split '{split_key}'. "
                f"Available: {sorted(node_splits.keys())}"
            )
        labels = _mask_labels_to_node_split(labels, node_splits[split_key])
        task = midterm_task(
            labels=labels,
            num_classes=num_classes,
            split=split,
            label_set=label_set,
            split_labels=False,
            train_cap=train_cap,
            linear_probe=linear_probe,
            random_query=kwargs.get("eval_random_query", False),
        )
        task.original_graph_labels = graph.y.numpy().copy()
        task.split_masked_labels = labels.copy()
        task_base = task
        if task_name == "cls_nm":
            counts = [int(v) for v in str(kwargs.get("cls_nm_task_counts", "1,1")).split(",")]
            if len(counts) != 2 or min(counts) < 1:
                raise ValueError("cls_nm_task_counts must be positive MT,NM integers")
            task_base = MultiTaskSplitBatch(
                [task, NeighborTask(positive_sampler_for_split(dataset, split, kwargs), graph.num_nodes,
                                    "inout", kwargs.get("neighbor_sampling_strategy", "strict"),
                                    filter_min_degree=bool(kwargs.get("neighbor_matching_edge_split", False)))],
                ["mct", "nt"], counts,
            )
            label_embeddings = {"mct": label_embeddings,
                                "nt": torch.zeros(1, 768).expand(graph.num_nodes, -1)}
        sampler = BatchSampler(
            batch_count,
            task_base,
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
        is_multiway = True
    elif task_name == "regression":
        if getattr(graph, "label_type", "classification") != "regression":
            raise ValueError(
                f"ukr_rus_twitter regression requires a graph with regression labels; "
                f"got label_type={getattr(graph, 'label_type', None)!r}."
            )
        if n_way != 1:
            raise ValueError(f"ukr_rus_twitter regression only supports n_way=1, got {n_way}.")

        labels = graph.y.detach().cpu().numpy().astype(np.float32)
        if not hasattr(dataset, "_regression_node_splits"):
            dataset._regression_node_splits = _build_regression_node_splits(labels, seed=0)

        split_key = (node_split or "").strip() or split
        node_splits = dataset._regression_node_splits
        if split_key not in node_splits:
            raise ValueError(
                f"Unknown ukr_rus_twitter regression node split '{split_key}'. "
                f"Available: {sorted(node_splits.keys())}"
            )
        split_idx = node_splits[split_key]
        if split_idx.size == 0:
            raise ValueError(
                f"ukr_rus_twitter regression split '{split_key}' has no labeled nodes."
            )

        task = RegressionTask(labels=labels, node_indices=split_idx)
        task.original_graph_labels = labels.copy()
        task.split_masked_labels = labels.copy()
        sampler = BatchSampler(
            batch_count,
            task,
            ParamSampler(batch_size, 1, n_shot, n_query, 1),
            seed=seed,
        )
        label_embeddings = torch.zeros((1, 768), dtype=torch.float)
        is_multiway = False
    else:
        raise ValueError(f"Unknown task for ukr_rus_twitter: {task_name}")

    aug_fn = get_aug(aug, graph.x) if (split == "train" or aug_test) else get_aug("")
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=Collator(label_embeddings, aug=aug_fn, is_multiway=is_multiway, task_name=task_name),
    )
