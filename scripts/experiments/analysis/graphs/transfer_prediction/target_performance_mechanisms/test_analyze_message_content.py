import unittest

from .analyze_message_content import summarize


class MessageAnalysisTest(unittest.TestCase):
    def test_each_condition_uses_its_own_model_stream_baseline(self):
        rows = []
        for seed, baseline in ((0, .8), (1, .6)):
            for condition, shift in (("intact", 0), ("actual_mean", .1)):
                rows.append({"target": "target", "stream": "original", "model_id": f"m{seed}", "source": "source", "seed": seed,
                             "condition": condition, "role": "both" if condition == "intact" else "support",
                             "roc_auc": baseline+shift, "accuracy": baseline+shift, "f1": baseline+shift, "nll": 1-baseline-shift})
        paired, summary = summarize(rows[::-1])
        effect = next(r for r in summary if r["condition"] == "actual_mean")
        self.assertEqual(effect["seed_streams"], 2)
        self.assertAlmostEqual(effect["roc_auc_mean"], .1)
        self.assertAlmostEqual(effect["nll_mean"], -.1)
        self.assertEqual(effect["roc_auc_positive"], 2)
        self.assertEqual(len(paired), 4)


if __name__ == "__main__":
    unittest.main()
