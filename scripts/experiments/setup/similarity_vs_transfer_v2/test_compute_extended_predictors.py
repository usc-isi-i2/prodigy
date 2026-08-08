from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


PATH = Path(__file__).with_name("compute_extended_predictors_tucker.py")
SPEC = importlib.util.spec_from_file_location("compute_extended_predictors_tucker", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ExtendedPredictorTest(unittest.TestCase):
    def test_numeric_user_ids_match_across_storage_types(self):
        integer = MODULE.hash_ids(np.asarray([123, 456], dtype=np.int64))
        text = MODULE.hash_ids(["123", "456"])
        np.testing.assert_array_equal(integer, text)

    def test_skew_summary_reports_left_and_right(self):
        rows = np.asarray([[0, 0], [0, 1], [0, 2], [10, 3]], dtype=float)
        result = MODULE.skew_summary(rows)
        self.assertEqual(result["n_dimensions"], 2)
        self.assertGreaterEqual(result["right_skew_fraction"], 0.5)

    def test_local_structure_signature_shape_and_finiteness(self):
        adjacency = csr_matrix(np.asarray([
            [0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]
        ]))
        result = MODULE.local_structure_signatures(
            adjacency, np.arange(4), fanout=10, rng=np.random.default_rng(1)
        )
        self.assertEqual(result.shape, (4, 17))
        self.assertTrue(np.isfinite(result).all())

    def test_shared_projection_preserves_graph_rows(self):
        samples = {"a": np.eye(4), "b": np.ones((3, 4))}
        projected = MODULE.shared_projection(samples, dims=2, seed=1)
        self.assertEqual(projected["a"].shape, (4, 2))
        self.assertEqual(projected["b"].shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
