import unittest

import numpy as np

from scripts.experiments.setup.ogbl_collab_mlp_lp.inspect_errors import (
    hits_threshold,
    select_examples,
)


class InspectErrorsTest(unittest.TestCase):
    def test_hits_threshold_is_kth_largest(self):
        self.assertEqual(hits_threshold(np.array([1.0, 4.0, 2.0, 3.0]), 2), 3.0)

    def test_selects_deepest_positive_misses(self):
        rows = [
            {"kind": "test_positive_false_negative", "stratum": "novel_vs_train", "mean_margin": -1.0},
            {"kind": "test_positive_false_negative", "stratum": "novel_vs_train", "mean_margin": -3.0},
            {"kind": "test_positive_false_negative", "stratum": "seen_in_train", "mean_margin": -4.0},
        ]
        selected = select_examples(rows, 1, "test_positive_false_negative", "novel_vs_train")
        self.assertEqual(selected[0]["mean_margin"], -3.0)

    def test_selects_highest_negative_false_positives(self):
        rows = [
            {"kind": "official_negative_false_positive", "stratum": "official_negative", "mean_margin": 1.0},
            {"kind": "official_negative_false_positive", "stratum": "official_negative", "mean_margin": 3.0},
        ]
        selected = select_examples(rows, 1, "official_negative_false_positive")
        self.assertEqual(selected[0]["mean_margin"], 3.0)


if __name__ == "__main__":
    unittest.main()
