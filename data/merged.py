"""
Loader for the merged disjoint-union graph produced by rapids-experiments/merge-graphs.

The .pt file is expected to contain all standard fields (x, edge_index, edge_attr,
y, edge_index_views, target_edge_index_views, …) plus:
  train_mask  bool [N]  — 80 % of nodes per dataset/label stratum
  test_mask   bool [N]  — remaining 20 %
  graph_id    long [N]  — which source dataset each node came from (0, 1, 2, …)
  dataset_info  list    — per-source metadata (name, offset, n_nodes, label_names)

Train/val episodes are sampled from train_mask nodes; test episodes from test_mask.
"""
import os
from typing import Optional, Set, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from experiments.sampler import NeighborSampler
from .augment import get_aug
from .dataloader import ParamSampler, BatchSampler, Collator, NeighborTask
from .dataset import SubgraphDataset
from .midterm import (
    _build_midterm_graph,
    _mask_labels_to_node_split,
    _deterministic_label_embeddings,
    midterm_task,
    BinaryFutureLinkTask,
    _normalize_view_name,
    _load_named_tensor,
)


# ---------------------------------------------------------------------------
# Masked task variants — restrict episode center selection to a node subset
# ---------------------------------------------------------------------------

class MaskedNeighborTask(NeighborTask):
    """NeighborTask whose centers are drawn only from allowed_indices."""

    def __init__(self, neighbor_sampler, allowed_indices: list, direction: str,
                 sampling_strategy: str = "strict"):
        # Pass full graph size to parent so subgraph fetching works correctly;
        # override center selection via allowed_indices.
        super().__init__(neighbor_sampler, len(allowed_indices), direction, sampling_strategy)
        self.allowed_indices = allowed_indices

    def sample(self, num_label, num_member, num_shot, num_query, rng):
        task = {}
        while len(task) < num_label:
            center = rng.choice(self.allowed_indices)
            if center in task:
                continue
            node_idx = torch.ones(num_member * 10, dtype=torch.long) * center
            node_idx = self.neighbor_sampler.random_walk(node_idx, self.direction)
            if node_idx.numel() == 0:
                continue
            unique_node_idx = torch.unique(node_idx)
            if unique_node_idx.size(0) >= num_member:
                task[center] = unique_node_idx[:num_member].tolist()
            elif self.sampling_strategy == "replacement":
                sampled = unique_node_idx.tolist()
                while len(sampled) < num_member:
                    sampled.append(rng.choice(sampled))
                task[center] = sampled[:num_member]
        return task


class MaskedBinaryFutureLinkTask(BinaryFutureLinkTask):
    """BinaryFutureLinkTask whose center nodes are drawn only from allowed_indices."""

    def __init__(self, future_neighbor_sampler, allowed_indices: list, size: int,
                 neg_ratio: int = 1):
        super().__init__(future_neighbor_sampler, size, neg_ratio)
        self.allowed_indices = allowed_indices

    def sample(self, num_label, num_member, num_shot, num_query, rng):
        del num_label, num_shot, num_query
        center = None
        retweeters = []
        for _ in range(2000):
            candidate = rng.choice(self.allowed_indices)
            curr = self._future_retweeters(candidate)
            if len(curr) >= num_member:
                center = candidate
                retweeters = curr
                break
        if center is None:
            raise RuntimeError(
                f"MaskedBinaryFutureLinkTask could not find a center with >= {num_member} "
                "future retweeters in the allowed node subset. "
                "Try reducing n_shots or n_query, or check that train/test masks contain "
                "nodes with sufficient future edges."
            )

        pos = rng.sample(retweeters, num_member)
        neg_target = num_member * self.neg_ratio
        forbidden = set(retweeters)
        forbidden.add(center)
        neg = []
        trials = 0
        max_trials = max(100, neg_target * 100)
        while len(neg) < neg_target and trials < max_trials:
            cand = rng.randrange(self.size)   # negatives still drawn from full graph
            if cand not in forbidden:
                neg.append(cand)
            trials += 1
        if len(neg) < neg_target:
            remaining = [n for n in self._all_nodes if n not in forbidden]
            if not remaining:
                raise RuntimeError("MaskedBinaryFutureLinkTask found no valid negative candidates.")
            while len(neg) < neg_target:
                neg.append(remaining[len(neg) % len(remaining)])

        return {(0, center): neg, (1, center): pos}


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def get_merged_dataset(
        root: str,
        n_hop: int = 1,
        graph_filename: str = "merged_all.pt",
        **kwargs,
) -> SubgraphDataset:
    graph_path = os.path.join(root, graph_filename)
    print(f"Loading merged graph from {graph_path}...")
    raw = torch.load(graph_path, map_location="cpu", weights_only=False)

    if "train_mask" not in raw or "test_mask" not in raw:
        raise ValueError(
            f"The merged .pt file at {graph_path} does not contain train_mask / test_mask. "
            "Re-run merge-graphs (rapids-experiments) to regenerate it with --test-ratio."
        )

    graph, resolved_edge_view = _build_midterm_graph(raw, **kwargs)

    print(f"Merged graph: {graph.num_nodes:,} nodes, {graph.edge_index.shape[1]:,} edges")
    labeled = (graph.y >= 0).sum().item()
    print(f"Labeled nodes: {labeled:,} / {graph.num_nodes:,} ({100 * labeled / graph.num_nodes:.1f}%)")

    train_mask = raw["train_mask"]
    test_mask  = raw["test_mask"]
    print(f"Split: train={train_mask.sum().item():,}  test={test_mask.sum().item():,}")

    dataset_info = raw.get("dataset_info", [])
    for info in dataset_info:
        ds_mask  = raw["graph_id"] == dataset_info.index(info)
        ds_train = (train_mask & ds_mask).sum().item()
        ds_test  = (test_mask  & ds_mask).sum().item()
        print(f"  [{info['name']}]  train={ds_train:,}  test={ds_test:,}  labels={info['label_names']}")

    print("Building neighbor sampler (CSR preprocessing)...", flush=True)
    neighbor_sampler = NeighborSampler(graph, num_hops=n_hop)
    print("Neighbor sampler ready.", flush=True)

    dataset = SubgraphDataset(graph, neighbor_sampler, bidirectional=False)
    dataset.train_mask = train_mask
    dataset.test_mask  = test_mask

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


# ---------------------------------------------------------------------------
# Dataloader builder
# ---------------------------------------------------------------------------

def get_merged_dataloader(
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
        **kwargs,
) -> DataLoader:
    del root
    task_name = kwargs.get("task_name", "neighbor_matching")
    seed = sum(ord(c) for c in split)

    graph = dataset.graph

    # Build allowed node index lists from the precomputed masks.
    # Val episodes reuse train nodes (same as midterm convention).
    train_indices = dataset.train_mask.nonzero(as_tuple=True)[0].tolist()
    test_indices  = dataset.test_mask.nonzero(as_tuple=True)[0].tolist()
    allowed = train_indices if split in ("train", "val") else test_indices

    if task_name == "neighbor_matching":
        task = MaskedNeighborTask(
            dataset.neighbor_sampler,
            allowed,
            "inout",
            kwargs.get("neighbor_sampling_strategy", "strict"),
        )
        label_embeddings = torch.zeros(1, 768).expand(graph.num_nodes, -1)
        sampler = BatchSampler(
            batch_count,
            task,
            ParamSampler(batch_size, n_way, n_shot, n_query, 1),
            seed=seed,
        )

    elif task_name == "classification":
        label_names = list(getattr(graph, "label_names", []))
        if not label_names:
            # Merged graph has per-dataset label names — use a generic fallback.
            n_classes = int((graph.y[graph.y >= 0].max().item() if (graph.y >= 0).any() else 1) + 1)
            label_names = [str(i) for i in range(n_classes)]

        if bert is not None:
            label_embeddings = bert.get_sentence_embeddings(label_names)
        else:
            label_embeddings = _deterministic_label_embeddings(label_names, dim=768)

        labels = graph.y.numpy()
        masked_labels = _mask_labels_to_node_split(labels, np.array(allowed))
        num_classes = len(label_names)

        task = midterm_task(
            labels=masked_labels,
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
                "Ensure --midterm_target_edge_view is set and the merged .pt has target_edge_index_views."
            )
        neg_ratio = int(kwargs.get("midterm_lp_neg_ratio", 1))
        task = MaskedBinaryFutureLinkTask(
            dataset.future_neighbor_sampler,
            allowed,
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
        raise ValueError(f"Unknown task for merged dataset: {task_name}")

    aug_fn = get_aug(aug, dataset.graph.x) if (split == "train" or aug_test) else get_aug("")
    is_multiway = task_name != "temporal_link_prediction"

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=Collator(label_embeddings, aug=aug_fn, is_multiway=is_multiway),
    )
