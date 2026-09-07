from itertools import product
import unittest

import pandas as pd

from .analyze_support_path import audit_paths, paired_effects, METRICS, STREAMS


class SupportPathTest(unittest.TestCase):
    def test_invariance_failure_and_missing_batch_rejected(self):
        ids = [f"model{i}" for i in range(6)]
        records = [dict(stream=s, model_id=m, batch=b, query_pre_and_post_bit_exact=True,
                        label_only_recomposition_bit_exact=True, label_vectors=8, changed_label_vectors=8,
                        label_max_abs_change=.1, mean_label_cosine=.9) for s, m, b in product(STREAMS, ids, range(32))]
        audit = pd.DataFrame(records)
        audit_paths(audit, ids)
        with self.assertRaises(ValueError):
            audit_paths(audit.iloc[1:], ids)
        audit.loc[0, "query_pre_and_post_bit_exact"] = False
        with self.assertRaisesRegex(ValueError, "invariance"):
            audit_paths(audit, ids)

    def test_effects_retain_failed_seed(self):
        rows = []
        for stream, seed, source, condition in product(STREAMS, range(3), ("cp_hk", "ukr_rus"), ("baseline", "edges_support")):
            score = .8 if source == "ukr_rus" else .6
            if condition == "edges_support" and source == "cp_hk":
                score += .1 if seed != 2 else -.1
            rows.append(dict(stream=stream, seed=seed, source=source, model_id=f"{source}_{seed}", condition=condition,
                             **{k: score for k in METRICS}))
        effects, gaps = paired_effects(pd.DataFrame(rows))
        self.assertEqual(len(effects), 12)
        self.assertTrue(effects[effects.source.eq("cp_hk") & effects.seed.eq(2)].delta_roc_auc.lt(0).all())
        self.assertTrue(gaps[gaps.seed.eq(2)].fraction_gap_reduced.lt(0).all())
        self.assertAlmostEqual(gaps[gaps.seed.eq(0)].fraction_gap_reduced.iloc[0], .5)


if __name__ == "__main__":
    unittest.main()
