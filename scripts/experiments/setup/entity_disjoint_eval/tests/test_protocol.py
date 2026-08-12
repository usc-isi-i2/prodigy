from dataclasses import dataclass

import pytest
import torch

from scripts.experiments.setup.entity_disjoint_eval.protocol import (
    canonical_global_id,
    AllowedNodePositiveSampler,
    select_center_clean_batches,
)


@dataclass
class Params:
    batch_size: int


def test_canonical_global_id_accepts_raw_and_namespaced_numeric_values():
    assert canonical_global_id(123) == "123"
    assert canonical_global_id("000123") == "123"
    assert canonical_global_id("ukr_rus:123") == "123"
    with pytest.raises(ValueError):
        canonical_global_id("local-user-name")


def test_center_filter_preserves_order_and_returns_full_batches():
    candidates = [
        ([{1: [11]}, {2: [12]}, {3: [13]}], Params(batch_size=3)),
        ([{4: [14]}, {5: [15]}, {6: [16]}], Params(batch_size=3)),
    ]
    batches, stats = select_center_clean_batches(
        candidates, {2, 5}, episode_count=4, batch_size=2
    )
    assert [[next(iter(ep)) for ep in batch] for batch, _ in batches] == [[1, 3], [4, 6]]
    assert all(params.batch_size == 2 for _, params in batches)
    assert stats == {
        "candidate_episodes_scanned": 6,
        "candidate_episodes_rejected": 2,
        "candidate_episodes_accepted": 4,
    }


def test_center_filter_rejects_support_or_query_centers_too():
    batches, stats = select_center_clean_batches(
        [([{1: [11]}, {2: [12]}, {3: [13]}], Params(batch_size=3))],
        {12}, episode_count=2, batch_size=1,
    )
    assert [next(iter(batch[0])) for batch, _ in batches] == [1, 3]
    assert stats["candidate_episodes_rejected"] == 1


def test_center_filter_fails_when_candidate_pool_is_too_small():
    with pytest.raises(RuntimeError, match="only 1 clean episodes"):
        select_center_clean_batches(
            [([{1: [2]}, {3: [4]}], Params(batch_size=2))],
            {3}, episode_count=2, batch_size=1,
        )


def test_allowed_positive_sampler_filters_walk_outputs():
    class Base:
        whole_adj = object()

        def random_walk(self, node_idx, direction):
            del node_idx, direction
            return torch.tensor([1, 2, 3, 4])

    sampler = AllowedNodePositiveSampler(
        Base(), torch.tensor([False, True, False, True, False])
    )
    assert sampler.random_walk(torch.tensor([0]), "inout").tolist() == [1, 3]
