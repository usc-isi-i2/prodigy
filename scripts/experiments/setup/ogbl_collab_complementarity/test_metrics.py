"""Check strict Hits@50 ties and the cost of negative-tail promotion."""
import unittest

import numpy as np

from run import compare, threshold


class ComplementarityTests(unittest.TestCase):
    def test_strict_tie_and_zero_rescue(self):
        negatives = np.concatenate((np.ones(50), np.zeros(50)))
        result = compare(np.array([1., 2.]), negatives,
                         np.array([1., 2.]), negatives)
        self.assertEqual(threshold(negatives), 1.)
        self.assertEqual(result["aadc_hits_at_50"], .5)
        self.assertEqual(result["rescue_curve"][0]["delta_pp"], 0.)

    def test_union_can_overstate_combined_result(self):
        an = np.concatenate((np.ones(50), np.zeros(50)))
        result = compare(np.array([2., .5]), an,
                         np.array([0., 3.]), an[::-1].copy())
        self.assertEqual(result["union_hit_fraction_not_combined_metric"], 1.)
        row = result["rescue_curve"][-1]
        self.assertEqual(row["negative50_normalized"], 2.)
        self.assertEqual(row["recovered_positives"], 1)
        self.assertEqual(row["lost_positives"], 1)
        self.assertEqual(row["hits_at_50"], .5)


if __name__ == "__main__":
    unittest.main()
