import unittest

from .analyze_role_topology import conditional_effects
from .audit_role_interactions import contrasts


class InteractionAnalysisTest(unittest.TestCase):
    def test_conditional_contrasts_do_not_duplicate_deterministic_cells(self):
        rows = []
        for draw in range(3):
            for q in ("intact", "removed", "rewired"):
                for s in ("intact", "removed", "rewired"):
                    if draw and "rewired" not in (q, s):
                        continue
                    v = .7 - .1*(q != "intact") + .05*(s != "intact") + .2*(q != "intact" and s != "intact")
                    rows.append({"target": "target", "stream": "original", "source": "source", "model_id": "model", "seed": 0,
                                 "query_condition": q, "support_condition": s, "draw": draw, "roc_auc": v, "accuracy": v, "nll": 1-v})
        effects = conditional_effects(rows)
        self.assertEqual(len(effects), 4)
        self.assertEqual(sum(r["change"] == "removed" for r in effects), 1)
        for r in effects:
            self.assertAlmostEqual(r["roc_auc_query_only"], -.1)
            self.assertAlmostEqual(r["roc_auc_query_given_support_changed"], .1)
            self.assertAlmostEqual(r["roc_auc_interaction"], .2)

    def test_reaudit_rejects_incomplete_or_mismatched_cells(self):
        row = {"stream": "original", "target": "target", "source": "source", "foreign_source": "True",
               "checkpoint": "ckpt", "weights_sha256": "weight", "episode_fingerprint": "episode", "queries": "1", "episodes": "1",
               "roc_auc": ".7", "accuracy": ".5", "nll": ".6"}
        rows = [{**row, "variant": v} for v in ("baseline", "edges_query", "edges_support", "edges_both")]
        self.assertEqual(len(contrasts(rows)), 1)
        with self.assertRaises(ValueError):
            contrasts(rows[:-1])
        with self.assertRaises(ValueError):
            contrasts(rows + [rows[0]])
        rows[0]["weights_sha256"] = "different"
        with self.assertRaises(ValueError):
            contrasts(rows)


if __name__ == "__main__":
    unittest.main()
