import json
from pathlib import Path
import shutil
import tempfile
import unittest

import pandas as pd

from .analyze_consumed_conflicts import HERE, link_targets, validate


class CompletedAuditTests(unittest.TestCase):
    def test_complete_saved_grid_and_target_provenance(self):
        m, w, r = validate(HERE/"data/consumed_conflicts")
        self.assertEqual(r["distinct_recorded_streams"], 27)
        joined, deltas = link_targets(m, pd.read_csv(HERE/"data/member_replay_cells.csv"),
            json.loads((HERE/"data/member_training_verified/arms.json").read_text()))
        self.assertEqual((len(joined), len(deltas)), (240, 180))

    def test_missing_window_and_changed_rate_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"audit"
            shutil.copytree(HERE/"data/consumed_conflicts", root)
            original = pd.read_csv(root/"windows.csv")
            original.iloc[1:].to_csv(root/"windows.csv", index=False)
            with self.assertRaisesRegex(ValueError, "window grid"):
                validate(root)
            original.to_csv(root/"windows.csv", index=False)
            models = pd.read_csv(root/"models.csv")
            models.loc[0, "query_also_wrong_class_support_fraction"] += .1
            models.to_csv(root/"models.csv", index=False)
            with self.assertRaises(AssertionError):
                validate(root)

    def test_changed_or_missing_target_checkpoint_fails(self):
        m, _, _ = validate(HERE/"data/consumed_conflicts")
        targets = pd.read_csv(HERE/"data/member_replay_cells.csv")
        verified = json.loads((HERE/"data/member_training_verified/arms.json").read_text())
        with self.assertRaisesRegex(ValueError, "target grid"):
            link_targets(m, targets.iloc[1:], verified)
        targets.loc[0, "weights_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "target checkpoint"):
            link_targets(m, targets, verified)


if __name__ == "__main__":
    unittest.main()
