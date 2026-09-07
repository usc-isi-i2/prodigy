import unittest
import numpy as np
from scripts.experiments.setup.target_performance_mechanisms.orientation_failure_risk import risk_summary


class RiskTests(unittest.TestCase):
    def calculate(self, margin, interaction, loss, weights=None, group=None):
        n = len(margin)
        return risk_summary(margin, interaction, loss,
                            ["equal"] * n if group is None else group,
                            np.ones(n) if weights is None else weights)

    def test_margin_confounding_removed(self):
        margin = np.repeat(np.arange(10), 4)
        interaction = margin * 10 + np.tile(np.arange(4), 10)
        r = self.calculate(margin, interaction, margin / 9)
        self.assertGreater(r["crude"]["risk_difference"], .4)
        for name in ("margin", "margin_degree"):
            self.assertAlmostEqual(r[name]["risk_difference"], 0)
            self.assertEqual(r[name]["coverage"], 1)

    def test_genuine_effect_survives(self):
        margin = np.repeat(np.arange(10), 4)
        interaction = np.tile(np.arange(4), 10)
        r = self.calculate(margin, interaction, (interaction > 1).astype(float))
        self.assertEqual(r["margin_degree"]["risk_difference"], 1)

    def test_degree_confounding_removed(self):
        group = ["equal"] * 4 + ["cue_correct"] * 4
        r = self.calculate(np.ones(8), np.arange(8), [0] * 4 + [1] * 4, group=group)
        self.assertEqual(r["margin"]["risk_difference"], 1)
        self.assertEqual(r["margin_degree"]["risk_difference"], 0)

    def test_ties_and_empty(self):
        for r in (self.calculate([1, 1], [2, 2], [0, 1]), self.calculate([], [], [])):
            self.assertEqual(r["crude"]["coverage"], 0)
            self.assertIsNone(r["crude"]["risk_difference"])
        row = self.calculate([1, 1, 1], [0, 1, 1], [0, .5, 1])["crude"]["strata"][0]
        self.assertEqual(row["low_n"], 3)

    def test_unequal_weights_and_standardization(self):
        r = self.calculate([1] * 4, [0, 1, 2, 3], [0, 1, 0, 1], [3, 1, 1, 1])
        self.assertAlmostEqual(r["crude"]["risk_difference"], 2 / 3)
        r = self.calculate([1] * 6, [0, 1, 0, 1, 2, 2], [0, 1, 1, 0, 1, 0],
                           [1, 1, 2, 2, 1, 1], ["equal"] * 2 + ["cue_correct"] * 2 + ["cue_wrong"] * 2)
        self.assertAlmostEqual(r["margin_degree"]["risk_difference"], -1 / 3)
        self.assertEqual(r["margin_degree"]["coverage"], .75)

    def test_validation(self):
        for args in (([1], [np.nan], [0]), ([1], [1], [2]), ([1], [], [0])):
            with self.assertRaises(ValueError):
                self.calculate(*args)
        with self.assertRaises(ValueError):
            self.calculate([1], [1], [0], [0])


if __name__ == "__main__":
    unittest.main()
