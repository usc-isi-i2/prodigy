from __future__ import annotations

import random

import pytest

from data.dataloader import BatchSampler, NeighborTask, ParamSampler


class IdentityWalkSampler:
    def random_walk(self, node_idx, direction):
        del direction
        return node_idx


def source_for_episode(episode: dict[int, list[int]]) -> int:
    sources = {center // 10 for center in episode}
    assert len(sources) == 1
    return sources.pop()


def test_complete_batch_has_one_within_source_episode_per_source() -> None:
    task = NeighborTask(
        IdentityWalkSampler(),
        size=30,
        direction="inout",
        strata=[range(0, 10), range(10, 20), range(20, 30)],
        confine_to_single_stratum=True,
        stratum_weighting="balanced",
        batch_source_mode="complete",
    )
    sampler = BatchSampler(
        num_samples=4,
        task=task,
        param_sampler=ParamSampler(3, 2, 0, 1, 1),
        seed=7,
    )

    for episodes, params in sampler:
        assert params.batch_size == 3
        assert sorted(source_for_episode(episode) for episode in episodes) == [0, 1, 2]


def test_complete_batch_rejects_wrong_batch_size() -> None:
    task = NeighborTask(
        IdentityWalkSampler(),
        size=20,
        direction="inout",
        strata=[range(0, 10), range(10, 20)],
        confine_to_single_stratum=True,
        batch_source_mode="complete",
    )
    sampler = BatchSampler(
        num_samples=1,
        task=task,
        param_sampler=ParamSampler(3, 2, 0, 1, 1),
        seed=7,
    )
    with pytest.raises(ValueError, match=r"number of active sources \(2\)"):
        sampler.sample()


def test_complete_batch_rejects_mixed_source_episodes() -> None:
    with pytest.raises(ValueError, match="cross_source_prob=0"):
        NeighborTask(
            IdentityWalkSampler(),
            size=20,
            direction="inout",
            strata=[range(0, 10), range(10, 20)],
            confine_to_single_stratum=True,
            cross_source_prob=0.1,
            batch_source_mode="complete",
        )
