import unittest

import pandas as pd

from .analyze_query_exchangeability import HERE, validate_frames


class QueryAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = [pd.read_csv(HERE/f"data/query_exchangeability/{name}.csv")
            for name in ("cells", "groups", "audits")]

    def test_complete_grid(self):
        validate_frames(*self.frames)

    def test_missing_cell_rejected(self):
        cells, groups, audits = self.frames
        with self.assertRaises(ValueError):
            validate_frames(cells.iloc[1:], groups, audits)

    def test_wrong_loss_decomposition_rejected(self):
        cells, groups, audits = self.frames
        changed = groups.copy()
        changed.loc[0, "within_group_imbalance_sum"] += .01
        with self.assertRaises(AssertionError):
            validate_frames(cells, changed, audits)

    def test_missing_identity_group_rejected(self):
        cells, groups, audits = self.frames
        with self.assertRaises(ValueError):
            validate_frames(cells, groups.iloc[1:], audits)

    def test_changed_frozen_query_predictions_rejected(self):
        cells, groups, audits = self.frames
        changed = audits.copy()
        changed.loc[changed["mode"] == "meta_frozen", "suffix_logit_error"] = .00001
        with self.assertRaises(ValueError):
            validate_frames(cells, groups, changed)


if __name__ == "__main__":
    unittest.main()
