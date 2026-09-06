import copy
import unittest

import pandas as pd

from .analyze_label_interface import CONTROLS, DECODERS, PANEL, contrasts, validate


def fixture():
    rows = []
    for target in PANEL:
        for source in range(9):
            for variant, decoders in [("baseline", DECODERS)] + [(v, ["full_model"]) for v in CONTROLS]:
                for decoder in decoders:
                    audit = {"forward_calls": 33, "training_label_count": 120, "original_ignore_label_embeddings": False,
                             "last_projected_mean_norm": 9., "last_interface_mean_norm": 0. if variant == "zero_projected_labels" else 16.,
                             "training_table_mean_norm": 16., "permutation_seed": 839113}
                    rows.append(dict(dataset=target, model_id=f"ss_{source}", variant=variant, decoder=decoder,
                                     episodes=128, queries=96, checkpoint=f"/{source}/state_dict_2500.ckpt",
                                     weights_sha256=f"weights-{source}", episode_fingerprint=target,
                                     roc_auc=.7 if variant == "baseline" else .71, accuracy=.6, f1=.5, nll=.8,
                                     label_interface_diagnostic=audit))
    cells = pd.DataFrame(rows)
    return cells, cells[cells.variant == "baseline"].copy(deep=True)


class LabelAnalysisTests(unittest.TestCase):
    def test_complete_grid_and_paired_contrast(self):
        cells, ref = fixture()
        checked = validate(cells, ref, "original")
        self.assertEqual(len(checked), 990)
        delta = contrasts(checked)
        self.assertEqual(len(delta), 225)
        self.assertTrue((abs(delta.delta_roc_auc - .01) < 1e-12).all())

    def test_missing_cell_and_changed_weights_rejected(self):
        cells, ref = fixture()
        with self.assertRaises(ValueError):
            validate(cells.iloc[1:], ref, "original")
        cells.loc[0, "weights_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate(cells, ref, "original")

    def test_baseline_or_control_norm_change_rejected(self):
        cells, ref = fixture()
        idx = cells.index[cells.variant == "baseline"][0]
        cells.loc[idx, "roc_auc"] = .8
        with self.assertRaisesRegex(ValueError, "baseline metric"):
            validate(cells, ref, "original")
        cells, ref = fixture()
        idx = cells.index[cells.variant == "train_norm_label_text"][0]
        cells.at[idx, "label_interface_diagnostic"] = copy.deepcopy(cells.at[idx, "label_interface_diagnostic"])
        cells.at[idx, "label_interface_diagnostic"]["last_interface_mean_norm"] = 9.
        with self.assertRaisesRegex(ValueError, "norm control"):
            validate(cells, ref, "original")


if __name__ == "__main__":
    unittest.main()
