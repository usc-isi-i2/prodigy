#!/usr/bin/env python3

import unittest

import numpy as np

from scripts.experiments.analysis.similarity_vs_transfer_v2.analyze_final_core_matrix import (
    permuted_asymmetry_stats,
    permuted_pairwise_stats,
    permuted_scalar_stats,
)
from scripts.experiments.analysis.similarity_vs_transfer_v2.analyze_predictors import (
    asymmetry_stat,
    scalar_matrix,
    within_target_stat,
)


class VectorizedPermutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(11)
        cls.outcome = rng.normal(size=(5, 5))
        cls.predictor = rng.normal(size=(5, 5))
        cls.values = rng.normal(size=5)
        cls.orders = np.stack([rng.permutation(5) for _ in range(30)])

    def test_pairwise_matches_serial_definition(self):
        expected = np.asarray([
            within_target_stat(self.predictor[np.ix_(order, order)], self.outcome)[0]
            for order in self.orders
        ])
        actual = permuted_pairwise_stats(self.predictor, self.outcome, self.orders)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_scalar_modes_match_serial_definition(self):
        for mode in ("source", "absolute_gap"):
            expected = np.asarray([
                within_target_stat(scalar_matrix(self.values[order], mode), self.outcome)[0]
                for order in self.orders
            ])
            actual = permuted_scalar_stats(self.values, self.outcome, mode, self.orders)
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_asymmetry_matches_serial_definition(self):
        expected = np.asarray([
            asymmetry_stat(self.values[order], self.outcome) for order in self.orders
        ])
        actual = permuted_asymmetry_stats(self.values, self.outcome, self.orders)
        np.testing.assert_allclose(actual, expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
