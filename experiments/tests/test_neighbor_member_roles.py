import random

import torch

from data.dataloader import NeighborTask


class FixedWalkSampler:
    def random_walk(self, node_idx, direction):
        del node_idx, direction
        # Deliberately unsorted with duplicates: torch.unique will return 1..8.
        return torch.tensor([8, 2, 6, 1, 5, 3, 7, 4, 8, 2])


def test_neighbor_members_are_shuffled_before_support_query_split():
    task = NeighborTask(FixedWalkSampler(), size=20, direction="inout")

    first = task._sample_center_members(0, 8, random.Random(1))
    second = task._sample_center_members(0, 8, random.Random(2))

    assert sorted(first) == list(range(1, 9))
    assert sorted(second) == list(range(1, 9))
    assert first != list(range(1, 9))
    assert second != list(range(1, 9))
    assert first != second


def test_disjoint_neighbor_members_are_also_shuffled():
    task = NeighborTask(FixedWalkSampler(), size=20, direction="inout")

    members = task._sample_center_members_disjoint(
        center=0,
        num_member=5,
        forbidden={2, 7},
        rng=random.Random(3),
    )

    assert len(members) == 5
    assert not ({2, 7} & set(members))
    assert members != sorted(members)
