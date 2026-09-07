import copy
import json
from pathlib import Path
import unittest

import pandas as pd

from .mixture_numerical_reference import numerical_reference, validate_numerical_audit


class NumericalReferenceTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[4] / "scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/data"
        self.audit = json.loads((root / "mixture_numerical_audit/audit.json").read_text())
        self.reference = pd.read_csv(root / "mixture_complementarity_inputs/classification_long.tsv", sep="\t")

    def test_exactly_one_auc_cell_changes_without_mutating_official_reference(self):
        original = self.reference.copy(deep=True)
        adjusted, change = numerical_reference(self.reference, self.audit)
        pd.testing.assert_frame_equal(self.reference, original)
        self.assertEqual(int((adjusted != original).to_numpy().sum()), 1)
        self.assertEqual(change["model_id"], "nmloo_without_cp_hk")
        self.assertEqual(change["dataset"], "ukr_rus_suspended")
        self.assertEqual(change["cpu_minus_official_roc_auc"], .5 / (128 * 128))

    def test_missing_repeat_changed_decision_non_tie_and_metric_drift_rejected(self):
        mutations = [
            lambda a: a["direct_forwards"].pop(),
            lambda a: a["direct_forwards"][0].update(changed_decisions=1),
            lambda a: a["direct_forwards"][0].update(max_abs_probability_difference=.001),
            lambda a: a["direct_forwards"][0]["metrics"].update(roc_auc=.52),
            lambda a: a["direct_forwards"][0]["changed_cross_label_pairs"][0].update(cpu_order=1),
            lambda a: a["direct_forwards"][0]["changed_cross_label_pairs"][0].update(other_scores=[.5, .51]),
            lambda a: a["cpu_prediction_audit"].update(weights_sha256="different"),
            lambda a: a["direct_forwards"][0]["metrics"].update(roc_auc=float("nan")),
            lambda a: a["direct_forwards"][0].update(positive_queries=127),
        ]
        for mutate in mutations:
            audit = copy.deepcopy(self.audit)
            mutate(audit)
            with self.assertRaises(ValueError):
                validate_numerical_audit(audit)
        changed = self.reference.copy()
        mask = (changed.model_id == "nmloo_without_cp_hk") & (changed.dataset == "ukr_rus_suspended")
        changed.loc[mask, "roc_auc"] = .6
        with self.assertRaisesRegex(ValueError, "immutable official"):
            numerical_reference(changed, self.audit)


if __name__ == "__main__":
    unittest.main()
