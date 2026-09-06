import unittest
import numpy as np
import pandas as pd

from .analyze_member_cue_alignment import cue_contrasts


class MemberCueTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for policy, value in zip(("lowest_sorted", "lowest_shuffled", "uniform_sorted", "uniform_shuffled"), (0., 1., 2., 5.)):
            rows.append({"stream": "original", "source": "cp_hk", "seed": 0,
                         "decoder": "full_model", "cue": "center_indegree", "policy": policy,
                         "valid_episodes": 128,
                         **{f"mean_within_episode_{m}": value for m in ("spearman", "pearson", "decision_agreement")}})
        return pd.DataFrame(rows)

    def test_fixed_factorial_coefficients(self):
        row = cue_contrasts(self.fixture()).iloc[0]
        self.assertEqual(row.retention_spearman, 3)
        self.assertEqual(row.role_shuffle_spearman, 2)
        self.assertEqual(row.interaction_spearman, 2)
        with self.assertRaises(ValueError):
            cue_contrasts(self.fixture().iloc[:3])

    def test_undefined_agreement_does_not_drop_an_arm(self):
        cells = self.fixture()
        cells.loc[0, "mean_within_episode_spearman"] = np.nan
        cells.loc[0, "valid_episodes"] = 0
        row = cue_contrasts(cells).iloc[0]
        self.assertTrue(np.isnan(row.retention_spearman))
        self.assertTrue(np.isnan(row.role_shuffle_spearman))
        self.assertEqual(row.minimum_valid_episodes, 0)


if __name__ == "__main__":
    unittest.main()
