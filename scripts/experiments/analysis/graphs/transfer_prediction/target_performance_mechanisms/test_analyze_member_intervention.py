import unittest
import pandas as pd
from .analyze_member_intervention import POLICIES, PRIMARY, factorial_contrasts, primary_contrast, verify_receipt


class FactorialAnalysisTests(unittest.TestCase):
    def test_known_main_effects_and_interaction(self):
        rows = []
        for source in ("cp_hk", "ukr_rus"):
            for target in PRIMARY:
                for seed in range(3):
                    for policy in POLICIES:
                        r = float(policy.startswith("uniform"))
                        o = float(policy.endswith("shuffled"))
                        y = .6 + (.06 if source == "cp_hk" else .02) * r + .01 * o + .02 * r * o
                        rows.append(dict(stream="original", source=source, seed=seed, dataset=target,
                                         decoder="full_model", policy=policy, roc_auc=y, accuracy=y, f1=y, nll=1-y))
        cells = pd.DataFrame(rows)
        contrasts = factorial_contrasts(cells)
        self.assertTrue(((contrasts.role_shuffle_roc_auc - .02).abs() < 1e-12).all())
        self.assertTrue(((contrasts.interaction_roc_auc - .02).abs() < 1e-12).all())
        primary = primary_contrast(contrasts)
        self.assertEqual(len(primary), 3)
        self.assertTrue(((primary.hong_kong_minus_ukraine_retention_effect - .04).abs() < 1e-12).all())
        with self.assertRaises(ValueError):
            factorial_contrasts(cells.iloc[1:])
        with self.assertRaises(ValueError):
            primary_contrast(contrasts[contrasts.dataset != "twibot20"])

    def test_smoke_receipt_cannot_be_research_evidence(self):
        with self.assertRaises(ValueError):
            verify_receipt({"valid": True, "research_result": False}, pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
