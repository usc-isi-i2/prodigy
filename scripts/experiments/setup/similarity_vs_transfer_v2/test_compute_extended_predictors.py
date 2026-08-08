from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


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


if __name__ == "__main__":
    unittest.main()
