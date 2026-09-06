import unittest

import pandas as pd

from .analyze_readout_training import SOURCES, paired_changes, primary_effect, validate_training
from .analyze_trajectories import DECODERS, PANEL


class ReadoutTrainingAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.cells = pd.DataFrame([
            dict(stream=stream, source=source, seed=seed, dataset=target, decoder=decoder,
                 condition=condition, episode_fingerprint=stream+target,
                 **{metric: .6 + (.05 if condition == "frozen" and not decoder.startswith("raw_") else 0)
                    for metric in ("roc_auc", "accuracy", "f1", "nll")})
            for stream in ("original", "fresh") for source in SOURCES for seed in range(3)
            for target in PANEL for decoder in DECODERS for condition in ("free", "frozen")])

    def test_complete_paired_grid_and_metric_sign(self):
        changes = paired_changes(self.cells)
        self.assertEqual(len(changes), 1530)
        self.assertAlmostEqual(changes.loc[changes.decoder == "full_model", "delta_roc_auc"].mean(), .05)
        with self.assertRaises(ValueError):
            paired_changes(self.cells.iloc[1:])
        bad = self.cells.copy()
        bad.loc[0, "episode_fingerprint"] = "other examples"
        with self.assertRaises(ValueError):
            paired_changes(bad)

    def test_raw_input_control_cannot_drift(self):
        bad = self.cells.copy()
        idx = bad.index[(bad.decoder == "raw_center/ridge") & (bad.condition == "frozen")][0]
        bad.loc[idx, "accuracy"] += .01
        with self.assertRaisesRegex(ValueError, "raw input"):
            paired_changes(bad)

    def test_primary_requires_both_streams_all_sources_and_each_seed(self):
        changes = paired_changes(self.cells)
        primary, supported = primary_effect(changes)
        self.assertEqual(len(primary), 6)
        self.assertTrue(supported)
        mask = (changes.dataset == "facebook_page_reference") & (changes.decoder == "full_model")
        changes.loc[mask & (changes.seed == 1), "delta_roc_auc"] = -.01
        self.assertFalse(primary_effect(changes)[1])
        with self.assertRaises(ValueError):
            primary_effect(changes[changes.source != "covid"])

    def test_smoke_and_missing_validity_receipts_rejected(self):
        for receipt in ({}, {"valid": True, "research_result": False, "models": 18}):
            with self.assertRaises(ValueError):
                validate_training(receipt, pd.DataFrame(), [])


if __name__ == "__main__":
    unittest.main()
