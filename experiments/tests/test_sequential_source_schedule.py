from __future__ import annotations

import random

import pytest
import torch
from torch_geometric.data import Data

from data.covid19_twitter import (
    get_covid19_twitter_dataloader,
    parse_source_sequence_steps,
    resolve_source_sequence,
)
from data.dataloader import BatchSampler, NeighborTask, ParamSampler


class IdentityWalkSampler:
    def random_walk(self, node_idx, direction):
        del direction
        return node_idx


class TinyMergedDataset:
    def __init__(self):
        self.graph = Data(
            x=torch.zeros(20, 4),
            graph_id=torch.tensor([0] * 10 + [1] * 10),
            num_nodes=20,
        )
        self.graph.source_graph_names = ["ukr_rus", "covid"]
        self.neighbor_sampler = IdentityWalkSampler()

    def __len__(self):
        return 20

    def __getitem__(self, index):
        return index


def sampled_stratum(task: NeighborTask, rng: random.Random) -> int:
    episode = task.sample(
        num_label=2,
        num_member=1,
        num_shot=0,
        num_query=1,
        rng=rng,
    )
    centers = list(episode)
    if all(0 <= center < 10 for center in centers):
        return 0
    if all(10 <= center < 20 for center in centers):
        return 1
    if all(20 <= center < 30 for center in centers):
        return 2
    raise AssertionError(f"episode crossed scheduled strata: {centers}")


def test_blocked_schedule_is_contiguous_and_exhausts_loudly() -> None:
    task = NeighborTask(
        IdentityWalkSampler(),
        size=30,
        direction="inout",
        strata=[range(0, 10), range(10, 20), range(20, 30)],
        confine_to_single_stratum=True,
        stratum_schedule_steps=[2, 3, 1],
    )
    rng = random.Random(7)
    assert [sampled_stratum(task, rng) for _ in range(6)] == [0, 0, 1, 1, 1, 2]
    with pytest.raises(RuntimeError, match="schedule exhausted"):
        sampled_stratum(task, rng)


def test_schedule_rejects_cross_source_mixing() -> None:
    with pytest.raises(ValueError, match="cross_source_prob=0"):
        NeighborTask(
            IdentityWalkSampler(),
            size=20,
            direction="inout",
            strata=[range(0, 10), range(10, 20)],
            confine_to_single_stratum=True,
            cross_source_prob=0.1,
            stratum_schedule_steps=[1, 1],
        )


def test_source_sequence_preserves_order_and_rejects_duplicates_or_omissions() -> None:
    names = ["ukr_rus", "covid", "midterm"]
    assert resolve_source_sequence("covid,ukr_rus,midterm", [0, 1, 2], names) == [1, 0, 2]
    with pytest.raises(ValueError, match="duplicate"):
        resolve_source_sequence("ukr_rus,covid,covid", [0, 1, 2], names)
    with pytest.raises(ValueError, match="every active source"):
        resolve_source_sequence("ukr_rus,covid", [0, 1, 2], names)


def test_source_sequence_steps_must_match_full_budget() -> None:
    assert parse_source_sequence_steps("3,2,2", 3, 7) == [3, 2, 2]
    with pytest.raises(ValueError, match="full training budget"):
        parse_source_sequence_steps("3,2,1", 3, 7)
    with pytest.raises(ValueError, match="positive"):
        parse_source_sequence_steps("3,0,4", 3, 7)


def test_schedule_does_not_consume_rng_for_source_selection() -> None:
    task = NeighborTask(
        IdentityWalkSampler(),
        size=20,
        direction="inout",
        strata=[range(0, 10), range(10, 20)],
        confine_to_single_stratum=True,
        stratum_schedule_steps=[1, 1],
    )
    rng = random.Random(11)
    expected = random.Random(11)
    # Center sampling consumes the RNG, but scheduled source selection itself must not.
    sampled_stratum(task, rng)
    reference_task = NeighborTask(
        IdentityWalkSampler(),
        size=10,
        direction="inout",
        strata=[range(0, 10)],
        confine_to_single_stratum=True,
        stratum_schedule_steps=[1],
    )
    sampled_stratum(reference_task, expected)
    assert rng.getstate() == expected.getstate()


def test_schedule_continues_across_epoch_iterators() -> None:
    task = NeighborTask(
        IdentityWalkSampler(),
        size=20,
        direction="inout",
        strata=[range(0, 10), range(10, 20)],
        confine_to_single_stratum=True,
        stratum_schedule_steps=[2, 2],
    )
    sampler = BatchSampler(
        num_samples=2,
        task=task,
        param_sampler=ParamSampler(1, 2, 0, 1, 1),
        seed=7,
    )

    first_epoch = list(iter(sampler))
    second_epoch = list(iter(sampler))

    def stratum(batch) -> int:
        centers = list(batch[0][0])
        return int(all(center >= 10 for center in centers))

    assert [stratum(batch) for batch in first_epoch] == [0, 0]
    assert [stratum(batch) for batch in second_epoch] == [1, 1]


def test_train_loader_validates_full_multi_epoch_budget() -> None:
    loader = get_covid19_twitter_dataloader(
        TinyMergedDataset(),
        split="train",
        node_split="",
        batch_size=1,
        n_way=2,
        n_shot=0,
        n_query=1,
        batch_count=2,
        root="",
        bert=None,
        num_workers=0,
        aug="",
        aug_test=False,
        split_labels=False,
        train_cap=None,
        linear_probe=False,
        task_name="neighbor_matching",
        epochs=2,
        neighbor_sampling_episode_source="graph_id",
        neighbor_sampling_source_subset="ukr_rus,covid",
        neighbor_sampling_source_sequence="ukr_rus,covid",
        neighbor_sampling_source_sequence_steps="2,2",
    )
    assert loader.batch_sampler.task.stratum_schedule_steps == [2, 2]

    with pytest.raises(ValueError, match="full training budget"):
        get_covid19_twitter_dataloader(
            TinyMergedDataset(),
            split="train",
            node_split="",
            batch_size=1,
            n_way=2,
            n_shot=0,
            n_query=1,
            batch_count=2,
            root="",
            bert=None,
            num_workers=0,
            aug="",
            aug_test=False,
            split_labels=False,
            train_cap=None,
            linear_probe=False,
            task_name="neighbor_matching",
            epochs=2,
            neighbor_sampling_episode_source="graph_id",
            neighbor_sampling_source_subset="ukr_rus,covid",
            neighbor_sampling_source_sequence="ukr_rus,covid",
            neighbor_sampling_source_sequence_steps="1,2",
        )
