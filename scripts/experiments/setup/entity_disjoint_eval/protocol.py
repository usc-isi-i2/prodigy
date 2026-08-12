"""Pure helpers and frozen constants for exact-ID-clean evaluation."""

from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
from pathlib import Path
from typing import Any, Iterable

import torch


TARGETS = ("ukr_rus", "covid", "midterm")
DB_TABLE = {
    "ukr_rus": "ids_ukraine",
    "covid": "ids_covid",
    "midterm": "ids_midterm",
}
CENTER_PROTOCOL = "final_core_exact_id_center_disjoint_union3_v1"
INDUCED_PROTOCOL = "final_core_exact_id_induced_disjoint_union3_v1"
PROTOCOL = CENTER_PROTOCOL
EPISODE_COUNT = 512


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_global_id(value: Any) -> str:
    """Normalize a Twitter snowflake without serializing it in outputs.

    Merged artifacts normally expose ``raw_user_ids``.  The namespaced fallback
    has the form ``source:id``; only a numeric suffix is accepted.
    """
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip()
    if ":" in text:
        suffix = text.rsplit(":", 1)[1].strip()
        if suffix.isdigit():
            text = suffix
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text.isdigit():
        raise ValueError("expected a non-negative numeric platform-global user ID")
    return str(int(text))


def select_center_clean_batches(
    candidate_batches: Iterable[tuple[list[dict[int, list[int]]], Any]],
    excluded_centers: set[int],
    *,
    episode_count: int,
    batch_size: int,
) -> tuple[list[tuple[list[dict[int, list[int]]], Any]], dict[str, int]]:
    """Filter candidate episodes by centers, then deterministically rebatch.

    The order of retained episodes is unchanged.  Every returned batch is full,
    which preserves the evaluator's fixed-size accounting.
    """
    if episode_count <= 0 or batch_size <= 0 or episode_count % batch_size:
        raise ValueError("episode_count must be positive and divisible by batch_size")
    accepted: list[dict[int, list[int]]] = []
    first_param = None
    candidates_seen = 0
    rejected = 0
    for episodes, params in candidate_batches:
        if first_param is None:
            first_param = params
        for episode in episodes:
            candidates_seen += 1
            referenced = [int(center) for center in episode]
            referenced.extend(int(member) for members in episode.values() for member in members)
            if any(node in excluded_centers for node in referenced):
                rejected += 1
                continue
            accepted.append(episode)
            if len(accepted) == episode_count:
                break
        if len(accepted) == episode_count:
            break
    if len(accepted) != episode_count or first_param is None:
        raise RuntimeError(
            f"only {len(accepted)} clean episodes after scanning {candidates_seen}; "
            f"need {episode_count}"
        )
    clean_param = replace(first_param, batch_size=batch_size)
    batches = [
        (accepted[start : start + batch_size], clean_param)
        for start in range(0, episode_count, batch_size)
    ]
    return batches, {
        "candidate_episodes_scanned": candidates_seen,
        "candidate_episodes_rejected": rejected,
        "candidate_episodes_accepted": episode_count,
    }


class AllowedNodePositiveSampler:
    """Filter positive-walk outputs while retaining the frozen holdout graph.

    This stage makes anchors and support/query centers entity-clean. Encoder
    subgraphs still use the original background sampler and are audited later.
    """

    def __init__(self, base_sampler: Any, allowed_mask: torch.Tensor):
        if allowed_mask.dtype != torch.bool or allowed_mask.ndim != 1:
            raise ValueError("allowed_mask must be a one-dimensional bool tensor")
        self.base_sampler = base_sampler
        self.allowed_mask = allowed_mask.cpu()
        self.whole_adj = base_sampler.whole_adj

    def random_walk(self, node_idx: torch.Tensor, direction: str) -> torch.Tensor:
        values = self.base_sampler.random_walk(node_idx, direction)
        if values.numel() == 0:
            return values
        return values[self.allowed_mask[values.detach().cpu()].to(values.device)]


def induced_neighbor_sampler(base_sampler: Any, allowed_mask: torch.Tensor) -> tuple[Any, dict[str, int]]:
    """Clone a sampler with the full-index induced adjacency on allowed nodes.

    Sparse values are preserved because they encode original edge IDs (and the
    reverse-edge sign convention) used by ``SubgraphDataset``.
    """
    if allowed_mask.dtype != torch.bool or allowed_mask.ndim != 1:
        raise ValueError("allowed_mask must be a one-dimensional bool tensor")
    from torch_sparse import SparseTensor

    whole_adj = base_sampler.whole_adj
    sizes = tuple(int(value) for value in whole_adj.sparse_sizes())
    if sizes != (allowed_mask.numel(), allowed_mask.numel()):
        raise ValueError(
            f"adjacency shape {sizes} does not match allowed mask length {allowed_mask.numel()}"
        )
    row, col, value = whole_adj.coo()
    allowed = allowed_mask.to(row.device)
    keep = allowed[row] & allowed[col]
    induced = SparseTensor(
        row=row[keep],
        col=col[keep],
        value=value[keep] if value is not None else None,
        sparse_sizes=sizes,
        is_sorted=True,
        trust_data=True,
    )
    cloned = copy.copy(base_sampler)
    cloned.whole_adj = induced
    metadata = {
        "original_adjacency_nnz": int(whole_adj.nnz()),
        "induced_adjacency_nnz": int(induced.nnz()),
    }
    del row, col, value, allowed, keep
    return cloned, metadata


def configure_allowed_episode_centers(loader: Any, allowed_indices: torch.Tensor) -> set[int]:
    """Restrict a final-core NM sampler to allowed anchors and members."""
    if allowed_indices.dtype != torch.long or allowed_indices.ndim != 1:
        raise ValueError("allowed_indices must be a one-dimensional long tensor")
    if allowed_indices.numel() < 30:
        raise RuntimeError("fewer than 30 allowed nodes cannot form a 30-way episode")
    task = loader.batch_sampler.task
    allowed_mask = torch.zeros(task.size, dtype=torch.bool)
    allowed_mask[allowed_indices] = True
    task.strata = [allowed_indices.tolist()]
    task.confine_to_single_stratum = True
    task.stratum_weights = [1.0]
    task._eligible_cache.clear()
    task.neighbor_sampler = AllowedNodePositiveSampler(task.neighbor_sampler, allowed_mask)
    return set(int(value) for value in allowed_indices.tolist())
