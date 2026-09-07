import unittest
import pandas as pd

from .analyze_trajectories import endpoint_changes, validate, PANEL, DECODERS, STEPS


class TrajectoryValidationTests(unittest.TestCase):
    def setUp(self):
        manifest, rows = [], []
        for s in range(9):
            for step in STEPS:
                name = f"ss_s{s}" + (f"__step{step}" if step != 2500 else "")
                path = f"/{s}/state_dict_{step}.ckpt"
                manifest.append(dict(model_id=name, checkpoint=path, sources=f"s{s}", source=f"s{s}", step=step))
                for target in PANEL:
                    for decoder in DECODERS:
                        rows.append(dict(dataset=target, model_id=name, checkpoint=path, sources=[f"s{s}"],
                            decoder=decoder, variant="baseline", episodes=128, queries=256,
                            episode_fingerprint=target, weights_sha256=name, roc_auc=.6, accuracy=.55, f1=.5,
                            nll=.7, official_metric_max_abs_error=0.))
        self.manifest = pd.DataFrame(manifest)
        self.cells = pd.DataFrame(rows)
        self.reference = self.cells[~self.cells.model_id.str.contains("__step")].copy()

    def check(self, cells):
        return validate(cells, self.manifest, self.reference, "original")

    def test_valid_and_no_checkpoint_selection(self):
        verified = self.check(self.cells)
        delta = endpoint_changes(verified)
        self.assertTrue((delta.change_100_to_2500 == 0).all())
        self.assertFalse(delta.all_three_increments_positive.any())

    def test_reject_missing_and_duplicate(self):
        for cells in (self.cells.iloc[1:], pd.concat([self.cells, self.cells.iloc[:1]])):
            with self.assertRaises(ValueError):
                self.check(cells)

    def test_reject_weight_input_and_metric_drift(self):
        for column, value in (("weights_sha256", "changed"), ("episode_fingerprint", "changed"),
                              ("checkpoint", "/wrong"), ("nll", float("nan"))):
            cells = self.cells.copy()
            cells.loc[0, column] = value
            with self.assertRaises(ValueError):
                self.check(cells)

    def test_reject_raw_probe_drift(self):
        cells = self.cells.copy()
        idx = cells.index[cells.decoder == "raw_center/ridge"][0]
        cells.loc[idx, "roc_auc"] = .8
        with self.assertRaisesRegex(ValueError, "raw probe"):
            self.check(cells)


if __name__ == "__main__":
    unittest.main()
