import unittest

from .analyze_support_dose import summarize


class DoseSummaryTest(unittest.TestCase):
    def test_matches_own_checkpoint_and_averages_draws_before_monotonicity(self):
        rows = []
        for step in (100, 2500):
            for dose in (0, 25, 50, 75, 100):
                if step == 100 and dose not in (0, 100):
                    continue
                for draw in (range(3) if dose not in (0, 100) else [0]):
                    value = .4 + step/10000 + dose/1000 + (draw-1)*.001 if dose not in (0, 100) else .4 + step/10000 + dose/1000
                    rows.append({"target": "t", "stream": "original", "model_id": "model", "source": "s", "seed": 0,
                                 "step": step, "suppression_percent": dose, "draw": draw,
                                 "roc_auc": value, "accuracy": value, "f1": value, "nll": 1-value})
        cells, steps, doses, curves = summarize(rows)
        self.assertEqual(len(cells), 13)
        self.assertEqual(len(steps), 2)
        self.assertEqual(len(doses), 5)
        self.assertEqual([r["draws"] for r in doses], [1, 3, 3, 3, 1])
        self.assertAlmostEqual(steps[0]["roc_auc_mean"], .1)
        self.assertAlmostEqual(steps[1]["roc_auc_mean"], .1)
        self.assertTrue(curves[0]["auc_nondecreasing"])
        self.assertFalse(curves[0]["auc_nonincreasing"])
        self.assertAlmostEqual(curves[0]["auc_delta_at_25"], .025)


if __name__ == "__main__":
    unittest.main()
