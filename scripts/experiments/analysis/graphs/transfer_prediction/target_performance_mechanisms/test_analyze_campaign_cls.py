import unittest

import pandas as pd

from .analyze_campaign_cls import DECODERS, PANEL, paired_deltas, validate


class CampaignAnalysisTests(unittest.TestCase):
    def complete_fixture(self):
        models = [f"model_{i}" for i in range(30)]
        manifest = pd.DataFrame([dict(model_id=m, checkpoint=f"/{m}/checkpoint", sources=["source"],
                                      forward_params={}, classification_adaptation={}) for m in models])
        inventory = pd.DataFrame([dict(model_id=m, weights_sha256=m) for m in models])
        cells = pd.DataFrame([dict(dataset=t, model_id=m, decoder=d, variant="baseline", episodes=128,
                                   queries=3072, checkpoint=f"/{m}/checkpoint", sources=["source"],
                                   weights_sha256=m, episode_fingerprint=t, roc_auc=.7, accuracy=.6, f1=.5, nll=.8)
                              for t in PANEL for m in models for d in DECODERS])
        return cells, manifest, inventory, cells[cells.model_id == models[0]].copy(deep=True)

    def test_complete_grid_is_required(self):
        cells, manifest, inventory, reference = self.complete_fixture()
        self.assertEqual(len(validate(cells, manifest, inventory, reference, "original")), 2550)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate(cells.iloc[1:], manifest, inventory, reference, "original")

    def test_checkpoint_input_and_raw_probe_changes_fail(self):
        for field, value, message in (("weights_sha256", "wrong", "weights"),
                                      ("checkpoint", "/wrong", "checkpoint"),
                                      ("episode_fingerprint", "wrong", "inputs")):
            cells, manifest, inventory, reference = self.complete_fixture()
            cells.loc[0, field] = value
            with self.assertRaisesRegex(ValueError, message):
                validate(cells, manifest, inventory, reference, "original")
        cells, manifest, inventory, reference = self.complete_fixture()
        cells.loc[cells.decoder == "raw_center/ridge", "roc_auc"] = .8
        with self.assertRaisesRegex(ValueError, "raw input probe"):
            validate(cells, manifest, inventory, reference, "original")

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
