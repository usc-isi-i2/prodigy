import unittest

import pandas as pd

from .analyze_campaign_cls import paired_deltas


class CampaignAnalysisTests(unittest.TestCase):
    def fixture(self):
        rows = []
        for role, baseline in (("common6000", .6), ("source_val_selected", .7)):
            for model, gain in (("nmi_baseline_r8_s0", 0), ("nmi_objective_r8_s0", .03)):
                rows.append(dict(stream="original", checkpoint_role=role, dataset="twibot20", decoder="full_model",
                                 original_model_id=model, roc_auc=baseline+gain, accuracy=baseline, f1=baseline, nll=1-baseline))
        return pd.DataFrame(rows)

    def test_baselines_are_specific_to_checkpoint_rule(self):
        d = paired_deltas(self.fixture())
        self.assertTrue((d[d.original_model_id == "nmi_baseline_r8_s0"].delta_roc_auc == 0).all())
        self.assertTrue((abs(d[d.original_model_id == "nmi_objective_r8_s0"].delta_roc_auc - .03) < 1e-12).all())

    def test_missing_or_duplicate_baseline_fails(self):
        cells = self.fixture()
        with self.assertRaises(ValueError):
            paired_deltas(cells.iloc[1:])
        with self.assertRaises(ValueError):
            paired_deltas(pd.concat([cells, cells.iloc[:1]], ignore_index=True))


if __name__ == "__main__":
    unittest.main()
