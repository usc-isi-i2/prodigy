"""Exact conditional member-law checks, not a whole-pipeline invariance claim."""
from collections import Counter
from itertools import permutations
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from data.dataloader import NeighborTask


def selection_law(policy, relabel=(0, 1, 2, 3), unordered=False):
    """Enumerate every equiprobable retention/role permutation on a four-ID set."""
    task = NeighborTask.__new__(NeighborTask)
    task.neighbor_sampler = SimpleNamespace(random_walk=lambda *args, **kwargs: torch.tensor(relabel))
    task.direction = "inout"
    task.member_policy = policy
    task.member_generators = {key: torch.Generator().manual_seed(3) for key in ("walk", "retention", "roles")}
    inverse = {renamed: original for original, renamed in enumerate(relabel)}
    retention = list(permutations(range(4))) if policy.startswith("uniform_") else [None]
    roles = list(permutations(range(2))) if policy.endswith("_shuffled") else [None]
    counts = Counter()
    for retained in retention:
        for role in roles:
            draws = [torch.tensor(x) for x in (retained, role) if x is not None]
            with patch("torch.randperm", side_effect=draws):
                chosen = task._sample_center_members(0, 2, random.Random(0))
            original_ids = tuple(inverse[x] for x in chosen)
            counts[tuple(sorted(original_ids)) if unordered else original_ids] += 1
    return counts


class MemberPermutationTests(unittest.TestCase):
    def test_historical_retained_set_changes_under_pure_relabeling(self):
        base = selection_law("lowest_sorted")
        renamed = selection_law("lowest_sorted", (3, 2, 1, 0))
        self.assertEqual(base, Counter({(0, 1): 1}))
        self.assertEqual(renamed, Counter({(3, 2): 1}))

    def test_role_shuffle_alone_does_not_remove_retention_bias(self):
        self.assertNotEqual(selection_law("lowest_shuffled", unordered=True),
                            selection_law("lowest_shuffled", (3, 2, 1, 0), unordered=True))

    def test_uniform_retention_alone_preserves_sets_but_not_role_law(self):
        self.assertEqual(selection_law("uniform_sorted", unordered=True),
                         selection_law("uniform_sorted", (3, 2, 1, 0), unordered=True))
        self.assertNotEqual(selection_law("uniform_sorted"),
                            selection_law("uniform_sorted", (3, 2, 1, 0)))

    def test_uniform_shuffled_ordered_law_is_exactly_equivariant(self):
        expected = Counter({pair: 4 for pair in permutations(range(4), 2)})
        self.assertEqual(selection_law("uniform_shuffled"), expected)
        for relabel in permutations(range(4)):
            self.assertEqual(selection_law("uniform_shuffled", relabel), expected)


if __name__ == "__main__":
    unittest.main()
