import unittest

import numpy as np

from .score_readout import score_task


class ScoreReadoutTests(unittest.TestCase):
    def test_paired_counts_and_f1(self):
        labels = np.array([0, 1, 2, 0])
        native = np.eye(3)[[0, 0, 2, 1]] * 4
        ridge = np.eye(3)[[0, 1, 1, 2]] * 4
        result = score_task(labels, native, ridge)
        self.assertEqual(result["counts"], {"both_correct": 1, "ridge_corrects_native": 1,
                                           "ridge_corrupts_native": 1, "both_wrong": 1})
        self.assertAlmostEqual(result["metrics"]["native"]["accuracy"], .5)
        self.assertAlmostEqual(result["metrics"]["native"]["macro_f1"], .5)
        self.assertAlmostEqual(result["metrics"]["ridge"]["macro_f1"], 4 / 9)
        self.assertFalse(result["examples"][3]["agreement"])
        for name in ("native", "ridge"):
            self.assertAlmostEqual(np.mean([x[f"{name}_nll"] for x in result["examples"]]),
                                   result["metrics"][name]["nll"])

    def test_column_permutation_preserves_metrics(self):
        labels = np.array([0, 1, 2])
        native = np.array([[3., 2., 1.], [1., 3., 2.], [3., 1., 2.]])
        ridge = native[:, ::-1].copy()
        original = score_task(labels, native, ridge)
        perm = np.array([2, 0, 1])
        changed = score_task(np.argsort(perm)[labels], native[:, perm], ridge[:, perm])
        self.assertEqual(original["counts"], changed["counts"])
        for name in ("native", "ridge"):
            for metric in original["metrics"][name]:
                self.assertAlmostEqual(original["metrics"][name][metric], changed["metrics"][name][metric])

    def test_rejects_invalid_inventory(self):
        with self.assertRaises(ValueError):
            score_task([0, 1], np.zeros((2, 2)), np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            score_task([0, 0], np.zeros((2, 2)), np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
