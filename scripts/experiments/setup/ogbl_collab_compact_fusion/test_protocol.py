import importlib.util
from pathlib import Path
import unittest

import numpy as np


PATH = Path(__file__).with_name("run.py")
SPEC = importlib.util.spec_from_file_location("compact_fusion", PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class FusionProtocolTest(unittest.TestCase):
    def test_parameter_cap_includes_both_experts(self):
        self.assertEqual(MOD.TOTAL_COUNTS["both"], 533_296)
        self.assertLess(max(MOD.TOTAL_COUNTS.values()), 1_000_000)

    def test_assessment_scores_use_frozen_scale(self):
        base = np.array([1.0, 2.0])
        expert = np.array([10.0, 20.0])
        actual = MOD.fuse(base, 2.0, expert, 10.0, expert, 10.0, 0.5, 0.0)
        expected = np.maximum(base / 2.0, 0.5 * np.exp(np.array([0.0, 10.0])))
        np.testing.assert_allclose(actual, expected)

    def test_zero_weights_are_exact_aadc_normalization(self):
        base = np.array([0.0, 2.0, 4.0])
        junk = np.array([100.0, -100.0, 3.0])
        np.testing.assert_array_equal(MOD.fuse(base, 2.0, junk, 1.0, junk, 1.0, 0.0, 0.0), base / 2.0)

    def test_candidate_grids(self):
        self.assertEqual(len(list(MOD.candidates("structure"))), len(MOD.ALPHAS))
        self.assertEqual(len(list(MOD.candidates("joint"))), len(MOD.ALPHAS))
        self.assertEqual(len(list(MOD.candidates("both"))), len(MOD.ALPHAS) ** 2)


if __name__ == "__main__":
    unittest.main()
