from __future__ import annotations

import random

import pytest
import torch
from torch_geometric.data import Data

from data.covid19_twitter import (
    get_covid19_twitter_dataloader,
    parse_source_sequence_steps,
    resolve_source_schedule,
    resolve_source_sequence,
)
from data.dataloader import BatchSampler, NeighborTask, ParamSampler
from data.dataset import SeededNodeIndex
from experiments.tests.test_shared_graph_training import tiny_dataset


class IdentityWalkSampler:
    def random_walk(self, node_idx, direction):
        del direction
        return node_idx


class SeededWalkSampler:
    class WholeAdj:
        @staticmethod
        def csr():
            return torch.arange(0, 301, 10), torch.arange(300) % 30, None

    whole_adj = WholeAdj()

    def random_walk(self, node_idx, direction, generator=None):
        del direction
        offsets = torch.randint(1, 30, node_idx.shape, generator=generator)
        return (node_idx + offsets) % 30


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


def test_repeated_schedule_revisits_strata_exactly() -> None:
    task = NeighborTask(
        IdentityWalkSampler(),
        size=30,
        direction="inout",
        strata=[range(0, 10), range(10, 20), range(20, 30)],
        confine_to_single_stratum=True,
        stratum_schedule=[1, 2, 0, 1, 2, 0],
        stratum_schedule_steps=[1, 2, 1, 1, 1, 2],
    )
    rng = random.Random(7)
    assert [sampled_stratum(task, rng) for _ in range(8)] == [1, 2, 2, 0, 1, 2, 0, 0]


def test_reordered_schedules_consume_identical_per_source_examples() -> None:
    def build(schedule):
        return NeighborTask(
            SeededWalkSampler(),
            size=20,
            direction="inout",
            strata=[range(0, 10), range(10, 20)],
            confine_to_single_stratum=True,
            stratum_schedule=schedule,
            stratum_schedule_steps=[1] * len(schedule),
            filter_min_degree=True,
            member_policy="uniform_shuffled",
            member_sampling_seed=71,
            source_schedule_seed=91,
        )

    by_source = []
    for schedule in ([0, 0, 1, 1], [0, 1, 0, 1]):
        task = build(schedule)
        streams = {0: [], 1: []}
        rng = random.Random(5)
        for source in schedule:
            streams[source].append(task.sample(2, 3, 1, 2, rng))
        by_source.append(streams)
    assert by_source[0] == by_source[1]
    assert all(
        isinstance(member, SeededNodeIndex)
        for episode in by_source[0].values()
        for task in episode
        for members in task.values()
        for member in members
    )


def test_reordered_schedules_expand_to_identical_full_contexts() -> None:
    def collect(schedule):
        dataset = tiny_dataset()
        task = NeighborTask(
            dataset.neighbor_sampler,
            size=24,
            direction="inout",
            strata=[range(0, 12), range(12, 24)],
            confine_to_single_stratum=True,
            stratum_schedule=schedule,
            stratum_schedule_steps=[1] * len(schedule),
            filter_min_degree=True,
            member_policy="uniform_shuffled",
            member_sampling_seed=71,
            source_schedule_seed=91,
        )
        streams = {0: [], 1: []}
        rng = random.Random(5)
        for source in schedule:
            episode = task.sample(2, 3, 1, 2, rng)
            contexts = []
            for members in episode.values():
                for member in members:
                    graph = dataset[member]
                    contexts.append(
                        (graph.global_node_ids.tolist(), graph.edge_index.tolist())
                    )
            streams[source].append(contexts)
        return streams

    torch.manual_seed(1234)
    state = torch.get_rng_state().clone()
    blocked = collect([0, 0, 1, 1])
    assert torch.equal(state, torch.get_rng_state())
    interleaved = collect([0, 1, 0, 1])
    assert blocked == interleaved


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


def test_source_schedule_allows_repeats_but_requires_complete_coverage() -> None:
    names = ["ukr_rus", "covid", "midterm"]
    assert resolve_source_schedule(
        "covid,ukr_rus,covid,midterm", [0, 1, 2], names
    ) == [1, 0, 1, 2]
    with pytest.raises(ValueError, match="at least once"):
        resolve_source_schedule("ukr_rus,covid,ukr_rus", [0, 1, 2], names)


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


def test_train_loader_maps_repeated_graph_ids_to_local_strata() -> None:
    loader = get_covid19_twitter_dataloader(
        TinyMergedDataset(),
        split="train",
        node_split="",
        batch_size=1,
        n_way=2,
        n_shot=0,
        n_query=1,
        batch_count=4,
        root="",
        bert=None,
        num_workers=0,
        aug="",
        aug_test=False,
        split_labels=False,
        train_cap=None,
        linear_probe=False,
        task_name="neighbor_matching",
        epochs=1,
        neighbor_sampling_episode_source="graph_id",
        neighbor_sampling_source_subset="ukr_rus,covid",
        neighbor_sampling_source_schedule="covid,ukr_rus,covid,ukr_rus",
        neighbor_sampling_source_schedule_steps="1,1,1,1",
    )
    task = loader.batch_sampler.task
    assert task.stratum_schedule == [1, 0, 1, 0]
    assert task.stratum_schedule_steps == [1, 1, 1, 1]
