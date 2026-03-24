import os
from typing import Dict, Optional, Set, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from experiments.sampler import NeighborSampler
from .augment import get_aug
from .dataloader import MulticlassTask, ParamSampler, BatchSampler, Collator, NeighborTask
from .dataset import SubgraphDataset


class BinaryFutureLinkTask:
    """
    Build true binary temporal LP episodes from a future-edge neighbor sampler.
    For each sampled center node:
    - positives: users that will retweet the center in the future (candidate -> center)
    - negatives: users that will not retweet the center in the future
    """
    def __init__(self, future_neighbor_sampler, size: int, neg_ratio: int = 1):
        self.future_neighbor_sampler = future_neighbor_sampler
        self.size = size
        self.neg_ratio = max(1, int(neg_ratio))
        self.rowptr, self.col, self.value = self.future_neighbor_sampler.whole_adj.csr()
        self._all_nodes = list(range(size))

    def _future_retweeters(self, center: int):
        start = int(self.rowptr[center].item())
        end = int(self.rowptr[center + 1].item())
        if end <= start:
            return []

        neigh = self.col[start:end]
        edge_ids = self.value[start:end]

        # preprocess(..., bidirectional=True) adds reverse edges with negative ids.
        # Row=center and negative id therefore means an original edge candidate -> center.
        incoming = neigh[edge_ids < 0].tolist()
        incoming = [int(n) for n in set(incoming) if int(n) != center]
        return incoming

    def sample(self, num_label, num_member, num_shot, num_query, rng):
        del num_label, num_shot, num_query
        center = None
        retweeters = []

        # Find a center with enough distinct future retweeters to fill support and
        # query without overlap (num_member = num_shot + num_query with n_aug=1).
        for _ in range(2000):
            candidate = rng.randrange(self.size)
            curr = self._future_retweeters(candidate)
            if len(curr) >= num_member:
                center = candidate
                retweeters = curr
                break
        if center is None:
            raise RuntimeError(
                f"BinaryFutureLinkTask could not find a center with >= {num_member} "
                "future retweeters. Try reducing n_shots or n_query."
            )

        # Positive samples without replacement so support and query nodes are distinct.
        pos = rng.sample(retweeters, num_member)
        neg_target = num_member * self.neg_ratio

        # Negative samples are users not in the future retweeter set (and not self).
        forbidden = set(retweeters)
        forbidden.add(center)
        neg = []
        trials = 0
        max_trials = max(100, neg_target * 100)
        while len(neg) < neg_target and trials < max_trials:
            cand = rng.randrange(self.size)
            if cand not in forbidden:
                neg.append(cand)
            trials += 1
        if len(neg) < neg_target:
            # Dense-graph fallback: deterministically fill from remaining node pool.
            remaining = [n for n in self._all_nodes if n not in forbidden]
            if not remaining:
                raise RuntimeError("BinaryFutureLinkTask found no valid negative candidates.")
            while len(neg) < neg_target:
                neg.append(remaining[len(neg) % len(remaining)])

        # Order matters for Collator(is_multiway=False): negatives first, positives second.
        return {(0, center): neg, (1, center): pos}


def _normalize_view_name(view_name: Optional[str], default: str = "default") -> str:
    name = (view_name or "").strip()
    return default if name == "" else name


def _load_named_tensor(
        raw: dict,
        view_name: str,
        *,
        default_key: str,
        views_key: str,
        legacy_prefix: str,
):
    if view_name == "default":
        return raw.get(default_key), "default"

    views = raw.get(views_key, {})
    if isinstance(views, dict) and view_name in views:
        return views[view_name], view_name

    legacy_key = f"{legacy_prefix}_{view_name}"
    if legacy_key in raw:
        return raw[legacy_key], view_name

    available = ["default"]
    if isinstance(views, dict):
        available.extend(sorted(views.keys()))
    available.extend(sorted(
        key[len(f"{legacy_prefix}_"):]
        for key in raw.keys()
        if key.startswith(f"{legacy_prefix}_")
    ))
    available = sorted(set(available))
    raise ValueError(
        f"Unknown view '{view_name}' for {default_key}. "
        f"Available views: {available}"
    )


def _load_edge_feature_names(raw: dict, resolved_edge_view: str):
    if resolved_edge_view == "default":
        names = raw.get("edge_attr_feature_names", [])
        return list(names) if isinstance(names, (list, tuple)) else []

    names_views = raw.get("edge_attr_feature_names_views", {})
    if isinstance(names_views, dict) and resolved_edge_view in names_views:
        names = names_views[resolved_edge_view]
        return list(names) if isinstance(names, (list, tuple)) else []

    legacy_key = f"edge_attr_feature_names_{resolved_edge_view}"
    names = raw.get(legacy_key, [])
    return list(names) if isinstance(names, (list, tuple)) else []


def _build_stratified_node_splits(
        labels: np.ndarray,
        *,
        seed: int = 0,
        train_frac: float = 0.6,
        val_frac: float = 0.2,
) -> Dict[str, np.ndarray]:
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    split_members = {"train": [], "val": [], "test": []}

    classes = sorted(int(c) for c in np.unique(labels) if int(c) >= 0)
    for cls in classes:
        idx = np.where(labels == cls)[0].copy()
        if idx.size == 0:
            continue
        rng.shuffle(idx)

        n = int(idx.size)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))

        if n >= 3:
            n_train = min(max(1, n_train), n - 2)
            n_val = min(max(1, n_val), n - n_train - 1)
        elif n == 2:
            n_train, n_val = 1, 0
        else:
            n_train, n_val = 1, 0

        n_test = n - n_train - n_val
        if n >= 3 and n_test <= 0:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            else:
                n_train -= 1

        split_members["train"].append(idx[:n_train])
        split_members["val"].append(idx[n_train:n_train + n_val])
        split_members["test"].append(idx[n_train + n_val:])

    return {
        key: (np.concatenate(vals) if len(vals) else np.empty(0, dtype=np.int64))
        for key, vals in split_members.items()
    }


def _mask_labels_to_node_split(labels: np.ndarray, allowed_idx: np.ndarray, off_value: int = -100) -> np.ndarray:
    masked = np.full_like(labels, off_value)
    if allowed_idx.size > 0:
        masked[allowed_idx] = labels[allowed_idx]
    return masked


def _apply_feature_subset(graph: Data, subset_spec: str) -> Data:
    spec = (subset_spec or "all").strip()
    if spec in {"", "all"}:
        print(f"Using all midterm features ({graph.x.shape[1]} dims).")
        return graph
    if spec == "constant1":
        graph.x = torch.ones((graph.x.shape[0], 1), dtype=graph.x.dtype, device=graph.x.device)
        graph.feature_names = ["const_1"]
        print("Using constant node feature: 1 dim (all ones).")
        return graph

    feature_names = list(getattr(graph, "feature_names", []))
    x_dim = graph.x.shape[1]
    if not feature_names or len(feature_names) != x_dim:
        feature_names = [f"f{i}" for i in range(x_dim)]

    emb_mask = [name.startswith("emb_") for name in feature_names]
    stats_idx = [i for i, is_emb in enumerate(emb_mask) if not is_emb]
    emb_idx = [i for i, is_emb in enumerate(emb_mask) if is_emb]
    label_names = list(getattr(graph, "label_names", []))
    n_label = max(1, len(label_names))

    def build_label_leak():
        leak = torch.zeros((graph.x.shape[0], n_label), dtype=graph.x.dtype, device=graph.x.device)
        y = getattr(graph, "y", None)
        if y is not None:
            labeled_mask = (y >= 0) & (y < n_label)
            if labeled_mask.any():
                leak[labeled_mask, y[labeled_mask].long()] = 1.0
        leak_names = [f"label_leak_{name}" for name in label_names]
        if not leak_names:
            leak_names = [f"label_leak_{i}" for i in range(n_label)]
        return leak, leak_names

    if spec == "label_only":
        leak, leak_names = build_label_leak()
        graph.x = leak
        graph.feature_names = leak_names
        print(
            "Applied midterm feature subset 'label_only': "
            f"{x_dim} -> {graph.x.shape[1]} dims."
        )
        return graph

    if spec == "emb_only_plus_label":
        if not emb_idx:
            raise ValueError("midterm_feature_subset='emb_only_plus_label' requires embedding features.")
        leak, leak_names = build_label_leak()
        graph.x = torch.cat([graph.x[:, emb_idx], leak], dim=1)
        graph.feature_names = [feature_names[i] for i in emb_idx] + leak_names
        print(
            "Applied midterm feature subset 'emb_only_plus_label': "
            f"{x_dim} -> {graph.x.shape[1]} dims."
        )
        return graph

    indices = None
    if spec == "stats_only":
        indices = stats_idx
    elif spec == "emb_only":
        indices = emb_idx
    elif spec.startswith("keep:"):
        keep = [s.strip() for s in spec.split(":", 1)[1].split(",") if s.strip()]
        keep_set = set(keep)
        indices = [i for i, name in enumerate(feature_names) if name in keep_set]
    elif spec.startswith("drop:"):
        drop = [s.strip() for s in spec.split(":", 1)[1].split(",") if s.strip()]
        drop_set = set(drop)
        indices = [i for i, name in enumerate(feature_names) if name not in drop_set]
    else:
        raise ValueError(
            f"Unsupported midterm_feature_subset='{subset_spec}'. "
            f"Use one of: all, constant1, stats_only, emb_only, emb_only_plus_label, label_only, keep:<...>, drop:<...>."
        )

    if not indices:
        raise ValueError(
            f"midterm_feature_subset='{subset_spec}' selected 0 columns out of {x_dim}."
        )

    graph.x = graph.x[:, indices]
    graph.feature_names = [feature_names[i] for i in indices]
    print(
        f"Applied midterm feature subset '{subset_spec}': "
        f"{x_dim} -> {graph.x.shape[1]} dims."
    )
    return graph


def _apply_edge_feature_subset(graph: Data, subset_spec: str, feature_names=None):
    if not hasattr(graph, "edge_attr") or graph.edge_attr is None:
        return graph

    spec = (subset_spec or "all").strip()
    if spec in {"", "all"}:
        names = list(feature_names or [])
        if names:
            print(f"Using all midterm edge features ({graph.edge_attr.shape[1]} dims): {names}")
        else:
            print(f"Using all midterm edge features ({graph.edge_attr.shape[1]} dims).")
        graph.edge_attr_feature_names = names
        return graph

    if spec == "none":
        graph.edge_attr = None
        graph.edge_attr_feature_names = []
        print("Disabled midterm edge features (midterm_edge_feature_subset='none').")
        return graph

    x_dim = graph.edge_attr.shape[1]
    names = list(feature_names or [])
    if not names or len(names) != x_dim:
        names = [f"ef{i}" for i in range(x_dim)]

    indices = None
    if spec.startswith("keep:"):
        keep = [s.strip() for s in spec.split(":", 1)[1].split(",") if s.strip()]
        keep_set = set(keep)
        indices = [i for i, name in enumerate(names) if name in keep_set]
    elif spec.startswith("drop:"):
        drop = [s.strip() for s in spec.split(":", 1)[1].split(",") if s.strip()]
        drop_set = set(drop)
        indices = [i for i, name in enumerate(names) if name not in drop_set]
    else:
        raise ValueError(
            f"Unsupported midterm_edge_feature_subset='{subset_spec}'. "
            "Use one of: all, none, keep:<...>, drop:<...>."
        )

    if not indices:
        raise ValueError(
            f"midterm_edge_feature_subset='{subset_spec}' selected 0 columns out of {x_dim}."
        )

    graph.edge_attr = graph.edge_attr[:, indices]
    graph.edge_attr_feature_names = [names[i] for i in indices]
    print(
        f"Applied midterm edge feature subset '{subset_spec}': "
        f"{x_dim} -> {graph.edge_attr.shape[1]} dims."
    )
    return graph


def _build_midterm_graph(raw: dict, **kwargs):
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

    graph = Data(
        x=raw['x'],
        edge_index=edge_index,
        y=raw['y'],
        num_nodes=raw['x'].shape[0],
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

    graph.label_names = raw['label_names']
    graph.feature_names = raw.get('feature_names', [])
    graph = _apply_feature_subset(graph, kwargs.get("midterm_feature_subset", "all"))
    graph = _apply_edge_feature_subset(
        graph,
        kwargs.get("midterm_edge_feature_subset", "all"),
        feature_names=edge_feature_names,
    )
    return graph, resolved_edge_view


def get_midterm_dataset(
        root: str,
        n_hop: int = 1,
        graph_filename: str = 'graph_data.pt',
        **kwargs
) -> SubgraphDataset:
    graph_path = os.path.join(root, graph_filename)
    print(f"Loading midterm graph from {graph_path}...")
    raw = torch.load(graph_path, map_location='cpu')

    graph, resolved_edge_view = _build_midterm_graph(raw, **kwargs)

    print(f"Graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges, "
          f"{len(graph.label_names)} state classes")
    labeled = (graph.y >= 0).sum().item()
    print(f"Labeled nodes: {labeled} / {graph.num_nodes} ({100 * labeled / graph.num_nodes:.1f}%)")

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
    del root
    task_name = kwargs.get("task_name", "neighbor_matching")
    seed = sum(ord(c) for c in split)

    graph = dataset.graph

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
        label_names = list(getattr(graph, 'label_names', []))
        num_classes = len(label_names)

        if bert is not None:
            label_embeddings = bert.get_sentence_embeddings(label_names)
        else:
            label_embeddings = torch.randn(num_classes, 768)

        labels = graph.y.numpy()
        if not hasattr(dataset, "_classification_node_splits"):
            dataset._classification_node_splits = _build_stratified_node_splits(labels, seed=0)

        split_key = (node_split or "").strip() or split
        node_splits = dataset._classification_node_splits
        if split_key not in node_splits:
            raise ValueError(
                f"Unknown midterm classification node split '{split_key}'. "
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
        )
        sampler = BatchSampler(
            batch_count,
            task,
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )
    elif task_name == "temporal_link_prediction":
        if not hasattr(dataset, "future_neighbor_sampler"):
            raise ValueError(
                "temporal_link_prediction requires target edges, but no future edge view was found. "
                "Provide --midterm_target_edge_view (or ensure 'future_edge_index' exists in graph_data.pt)."
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
    else:
        raise ValueError(f"Unknown task for midterm: {task_name}")

    aug_fn = get_aug(aug, dataset.graph.x) if (split == "train" or aug_test) else get_aug("")
    is_multiway = task_name != "temporal_link_prediction"
    episode_label_leak = bool(kwargs.get("midterm_episode_label_leak", False))
    if episode_label_leak:
        print("Using episode-local label leak in collator (sanity mode).")

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=Collator(
            label_embeddings,
            aug=aug_fn,
            is_multiway=is_multiway,
            episode_label_leak=episode_label_leak,
        ),
    )
