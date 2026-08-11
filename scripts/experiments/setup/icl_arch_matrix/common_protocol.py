#!/usr/bin/env python3
"""Shared data contract for the seed-0 PRODIGY/VISION/GILT comparison."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import yaml


CHECKPOINT_STEPS = (20, 60, 180, 500)
TRAIN_STEPS = 500
TRAIN_BATCH_SIZE = 4
EVAL_EPISODES = 128
EVAL_BATCH_SIZE = 4
EVAL_N_WAY = 2
EVAL_N_SHOT = 10
N_WAY = 30
N_SHOT = 3
N_QUERY = 4
EPISODE_RNG_SEED = 271828


@dataclass
class Episode:
    """One task, with PRODIGY's synthetic pooling nodes removed."""

    x: torch.Tensor
    edge_index: torch.Tensor
    centers: torch.Tensor
    labels: torch.Tensor
    query_mask: torch.Tensor
    global_centers: torch.Tensor
    label_map: torch.Tensor | None = None
    global_node_ids: torch.Tensor | None = None

    @property
    def support_mask(self) -> torch.Tensor:
        return ~self.query_mask

    @property
    def n_way(self) -> int:
        return int(self.labels.max().item()) + 1

    @property
    def n_shot(self) -> int:
        counts = torch.bincount(self.labels[self.support_mask], minlength=self.n_way)
        if counts.unique().numel() != 1:
            raise ValueError(f"unequal support counts: {counts.tolist()}")
        return int(counts[0])

    @property
    def n_query(self) -> int:
        counts = torch.bincount(self.labels[self.query_mask], minlength=self.n_way)
        if counts.unique().numel() != 1:
            raise ValueError(f"unequal query counts: {counts.tolist()}")
        return int(counts[0])


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    expected = {
        "n_way": N_WAY,
        "n_shots": N_SHOT,
        "n_query": N_QUERY,
        "batch_size": TRAIN_BATCH_SIZE,
    }
    for key, value in expected.items():
        if int(config[key]) != value:
            raise ValueError(f"common protocol requires {key}={value}, got {config[key]}")
    if config.get("edge_view") != "static_train":
        raise ValueError("message passing must use edge_view=static_train")
    if config.get("target_edge_view") != "static_test":
        raise ValueError("held-out evaluation positives must use target_edge_view=static_test")
    if not config.get("neighbor_matching_edge_split"):
        raise ValueError("neighbor_matching_edge_split must be enabled")
    return config


def build_dataset(config: dict):
    from data.covid19_twitter import get_covid19_twitter_dataset

    return get_covid19_twitter_dataset(
        root=config["root"],
        graph_filename=config["graph_filename"],
        n_hop=int(config["n_hop"]),
        task_name="neighbor_matching",
        edge_view=config["edge_view"],
        target_edge_view=config["target_edge_view"],
        neighbor_matching_edge_split=True,
        feature_subset=config.get("feature_subset", "all"),
        seed=0,
        neighbor_sampling_hop_sizes=config.get("neighbor_sampling_hop_sizes", "9,9"),
        neighbor_sampling_node_limit=int(config.get("neighbor_sampling_node_limit", 101)),
    )


def build_loader(
    dataset,
    config: dict,
    *,
    split: str,
    sources: str,
    batch_count: int,
    batch_size: int,
    workers: int = 0,
):
    from data.covid19_twitter import get_covid19_twitter_dataloader

    return get_covid19_twitter_dataloader(
        dataset,
        split=split,
        node_split="",
        batch_size=batch_size,
        n_way=N_WAY,
        n_shot=N_SHOT,
        n_query=N_QUERY,
        batch_count=batch_count,
        root=config["root"],
        bert=None,
        num_workers=workers,
        aug="",
        aug_test=False,
        split_labels=False,
        train_cap=None,
        linear_probe=False,
        task_name="neighbor_matching",
        neighbor_sampling_strategy="strict",
        neighbor_sampling_episode_source="graph_id",
        neighbor_sampling_episode_source_weighting="balanced",
        neighbor_sampling_batch_source_mode="independent",
        neighbor_sampling_source_subset=sources,
        neighbor_sampling_cross_source_prob=0.0,
        neighbor_matching_edge_split=True,
        epochs=1,
        seed=0,
    )


def query_mask_from_batch(batch, n_way: int = N_WAY) -> torch.Tensor:
    labels = batch[2]
    edge_mask = batch[5]
    if edge_mask.numel() != labels.size(0) * n_way:
        raise ValueError(
            f"bad metagraph mask: {edge_mask.numel()} values for "
            f"{labels.size(0)} samples and {n_way} labels"
        )
    reshaped = edge_mask.reshape(labels.size(0), n_way)
    if not torch.all(reshaped == reshaped[:, :1]):
        raise ValueError("query mask differs across a sample's label edges")
    return reshaped[:, 0].bool()


def iter_episodes(
    batch,
    *,
    n_way: int = N_WAY,
    n_shot: int = N_SHOT,
    n_query: int = N_QUERY,
    equal_query_counts: bool = True,
) -> Iterator[Episode]:
    graphs = batch[0]
    labels = batch[2].argmax(dim=1).long()
    query_mask = query_mask_from_batch(batch, n_way=n_way)
    task_ids = graphs.task_id_per_sample.long()
    ptr = graphs.ptr.long()
    for task_id in task_ids.unique(sorted=True).tolist():
        sample_ids = torch.where(task_ids == task_id)[0]
        expected = torch.arange(sample_ids[0], sample_ids[-1] + 1, device=sample_ids.device)
        if not torch.equal(sample_ids, expected):
            raise ValueError("task samples are not contiguous in the PyG batch")
        node_lo = int(ptr[sample_ids[0]])
        node_hi = int(ptr[sample_ids[-1] + 1])

        local_global_ids = graphs.global_node_ids[node_lo:node_hi]
        keep = local_global_ids >= 0  # discard PRODIGY's isolated pooling supernodes
        old_to_new = torch.full((node_hi - node_lo,), -1, dtype=torch.long, device=keep.device)
        old_to_new[keep] = torch.arange(int(keep.sum()), device=keep.device)

        edge_index = graphs.edge_index
        edge_keep = (
            (edge_index[0] >= node_lo)
            & (edge_index[0] < node_hi)
            & (edge_index[1] >= node_lo)
            & (edge_index[1] < node_hi)
        )
        local_edges_old = edge_index[:, edge_keep] - node_lo
        local_edges = old_to_new[local_edges_old]
        valid_edges = (local_edges >= 0).all(dim=0)
        local_edges = local_edges[:, valid_edges]

        center_old = ptr[sample_ids] - node_lo
        centers = old_to_new[center_old]
        if (centers < 0).any():
            raise ValueError("an episode center resolved to a synthetic pooling node")

        episode = Episode(
            x=graphs.x[node_lo:node_hi][keep],
            edge_index=local_edges,
            centers=centers,
            labels=labels[sample_ids],
            query_mask=query_mask[sample_ids],
            global_centers=local_global_ids[center_old],
            label_map=(
                graphs.task_label_map[task_id].long()
                if hasattr(graphs, "task_label_map")
                else torch.arange(n_way, device=labels.device)
            ),
            global_node_ids=local_global_ids[keep],
        )
        _validate_episode(
            episode,
            n_way=n_way,
            n_shot=n_shot,
            n_query=n_query,
            equal_query_counts=equal_query_counts,
        )
        yield episode


def _validate_episode(
    episode: Episode,
    *,
    n_way: int,
    n_shot: int,
    n_query: int,
    equal_query_counts: bool,
) -> None:
    expected_items = n_way * (n_shot + n_query)
    if episode.centers.numel() != expected_items:
        raise ValueError(f"expected {expected_items} samples, got {episode.centers.numel()}")
    if int(episode.support_mask.sum()) != n_way * n_shot:
        raise ValueError("wrong support count")
    if int(episode.query_mask.sum()) != n_way * n_query:
        raise ValueError("wrong query count")
    for label in range(n_way):
        label_mask = episode.labels == label
        if int((label_mask & episode.support_mask).sum()) != n_shot:
            raise ValueError(f"label {label} has the wrong support count")
        if equal_query_counts and int((label_mask & episode.query_mask).sum()) != n_query:
            raise ValueError(f"label {label} has the wrong query count")


def classification_targets(catalog_path: str | Path) -> dict[str, dict]:
    """Return the fixed four-graph classification panel from the graph catalog."""
    import json

    selected = {
        "covid_political",
        "election2020",
        "ukr_rus_suspended",
        "twibot20",
    }
    with open(catalog_path, encoding="utf-8") as handle:
        catalog = json.load(handle)
    targets = {}
    for graph in catalog["graphs"]:
        key = graph["dataset_key"]
        if key not in selected:
            continue
        eval_config = graph["eval"]
        targets[key] = {
            "relative_path": graph["relative_path"],
            "n_query": int(eval_config["pl_n_query"]),
            "eval_random_query": bool(eval_config.get("eval_random_query", False)),
        }
    missing = selected - targets.keys()
    if missing:
        raise ValueError(f"classification targets missing from catalog: {sorted(missing)}")
    return targets


def build_classification_dataset(
    *,
    dataset_name: str,
    data_root: str | Path,
    target: dict,
):
    """Build one downstream graph while retaining the repository's native loader."""
    if dataset_name == "covid_political":
        from data.covid_political import (
            get_covid_political_dataloader as get_dataloader,
            get_covid_political_dataset as get_dataset,
        )
    else:
        from data import social_llm_dataset

        get_dataset = getattr(social_llm_dataset, f"get_{dataset_name}_dataset")
        get_dataloader = getattr(social_llm_dataset, f"get_{dataset_name}_dataloader")

    graph_path = Path(data_root) / target["relative_path"]
    dataset = get_dataset(
        root=str(graph_path.parent),
        graph_filename=graph_path.name,
        n_hop=2,
        task_name="classification",
        feature_subset="emb_only",
        seed=0,
        neighbor_sampling_hop_sizes="9,9",
        neighbor_sampling_node_limit=101,
    )
    return dataset, get_dataloader, graph_path


def build_classification_loader(
    *,
    dataset_name: str,
    data_root: str | Path,
    target: dict,
    dataset=None,
    get_dataloader=None,
    graph_path=None,
    workers: int = 0,
):
    """Build the exact deterministic test episodes shared by all architectures."""
    if dataset is None:
        dataset, get_dataloader, graph_path = build_classification_dataset(
            dataset_name=dataset_name, data_root=data_root, target=target
        )
    return get_dataloader(
        dataset,
        split="test",
        node_split="",
        batch_size=EVAL_BATCH_SIZE,
        n_way=EVAL_N_WAY,
        n_shot=EVAL_N_SHOT,
        n_query=int(target["n_query"]),
        batch_count=EVAL_EPISODES // EVAL_BATCH_SIZE,
        root=str(graph_path.parent),
        bert=None,
        num_workers=workers,
        aug="",
        aug_test=False,
        split_labels=False,
        train_cap=None,
        linear_probe=False,
        task_name="classification",
        eval_random_query=target["eval_random_query"],
        eval_episode_seed_offset=0,
        seed=0,
    )


def update_episode_fingerprint(hasher, episode: Episode) -> None:
    if episode.label_map is None or episode.global_node_ids is None:
        raise ValueError("fingerprinting requires global label and sampled-node identities")
    for tensor in (
        episode.global_centers,
        episode.labels,
        episode.query_mask.long(),
        episode.label_map,
        episode.global_node_ids,
        episode.edge_index,
    ):
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())


def new_fingerprint():
    return hashlib.sha256()


def reset_episode_rng(seed: int = EPISODE_RNG_SEED) -> None:
    """Reset sampling RNG after architecture-specific model initialization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
