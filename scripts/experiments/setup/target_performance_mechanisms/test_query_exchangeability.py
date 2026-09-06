import math
import unittest

import numpy as np

from .query_exchangeability import cyclic_permutation, symmetrized_groups, group_totals


class ExchangeabilityTests(unittest.TestCase):
    def test_realized_perfect_predictions_can_exceed_symmetrized_ceiling(self):
        z = np.array([[10., 0., -2.], [0., 10., -2.]])
        total = group_totals(symmetrized_groups(z, np.array([0, 1]), np.array([9, 9]), np.array([0, 0])))
        self.assertEqual(total["original_accuracy"], 1)
        self.assertEqual(total["symmetrized_accuracy"], .5)
        self.assertLess(total["original_nll"], math.log(2))
        self.assertGreater(total["symmetrized_nll"], math.log(2))

    def test_exact_cyclic_enumeration_and_decomposition(self):
        rng = np.random.default_rng(319)
        for k in range(1, 9):
            z, y = rng.normal(size=(k, 12))*3, np.arange(k)
            ids, tasks = np.ones(k, dtype=int), np.zeros(k, dtype=int)
            row = group_totals(symmetrized_groups(z, y, ids, tasks))
            losses, accuracies = [], []
            for shift in range(k):
                moved = z[cyclic_permutation(ids, tasks, shift)]
                p = np.exp(moved-moved.max(axis=1, keepdims=True))
                p /= p.sum(axis=1, keepdims=True)
                losses.append(-np.log(p[np.arange(k), y]).mean())
                accuracies.append((moved.argmax(1) == y).mean())
            self.assertAlmostEqual(row["symmetrized_nll"], np.mean(losses))
            self.assertAlmostEqual(row["symmetrized_accuracy"], np.mean(accuracies))
            self.assertAlmostEqual(row["symmetrized_nll"], row["nll_floor"]+row["mass_deficit"]+row["within_group_imbalance"])

    def test_grouping_never_crosses_episode_and_plan_needs_no_labels(self):
        ids, tasks = np.array([2, 2, 2, 2, 3]), np.array([0, 1, 0, 1, 0])
        p = cyclic_permutation(ids, tasks)
        np.testing.assert_array_equal(p, [2, 3, 0, 1, 4])
        np.testing.assert_array_equal(ids[p], ids)
        np.testing.assert_array_equal(tasks[p], tasks)

    def test_same_class_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            symmetrized_groups(np.ones((2, 3)), np.array([0, 0]), np.array([1, 1]), np.array([0, 0]))

    def test_floor_achieved_with_uniform_mass_only_on_group_classes(self):
        z = np.array([[0., 0., -1000.], [0., 0., -1000.]])
        row = group_totals(symmetrized_groups(z, np.array([0, 1]), np.array([3, 3]), np.array([0, 0])))
        self.assertAlmostEqual(row["symmetrized_nll"], math.log(2))
        self.assertAlmostEqual(row["mass_deficit"], 0)
        self.assertAlmostEqual(row["within_group_imbalance"], 0)

    def test_query_count_and_cohort_conservation(self):
        z = np.arange(20).reshape(5, 4)
        rows = symmetrized_groups(z, np.array([0, 1, 2, 2, 3]), np.array([3, 3, 4, 4, 5]), np.array([0, 0, 0, 1, 1]))
        self.assertEqual(len(rows), 4)
        all_rows = group_totals(rows)
        self.assertEqual(all_rows["queries"], 5)
        self.assertAlmostEqual(all_rows["accuracy_ceiling"], .8)


if __name__ == "__main__":
    unittest.main()
