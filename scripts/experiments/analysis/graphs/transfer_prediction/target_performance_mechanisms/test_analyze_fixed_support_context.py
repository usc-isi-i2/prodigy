import copy
import hashlib
from itertools import product
import json
import unittest

import pandas as pd

from .analyze_fixed_support_context import validate_plans, validate_frames, summarize, PANEL, STREAMS, METRICS


def reseal(p):
    p["sha256"] = hashlib.sha256(json.dumps({k: v for k, v in p.items() if k != "sha256"}, sort_keys=True).encode()).hexdigest()


def tiny_plans():
    plans = []
    for s, t in product(STREAMS, PANEL):
        rows = []
        for draw in range(2):
            seed = None if draw == 0 else 96060000+(10000000 if s == "fresh" else 0)+sum((i+1)*ord(c) for i, c in enumerate(t))*1000+draw
            rows.append(dict(batch=0, draw=draw, seed=seed, support_centers=list(range(80)), input_sha256="a",
                query_graphs_bit_exact=True, support_centers_and_center_features_bit_exact=True, all_metagraph_and_truth_tensors_bit_exact=True))
        p = dict(stream=s, target=t, graph_file_predates_original_cache=True,
            every_cached_and_new_feature_and_edge_verified=True, query_truth_and_feature_tamper_passed=True,
            same_sampler=dict(hops=2, hop_sizes=[9, 9], node_limit=101), original=dict(batch_sha256=["a"]), rows=rows)
        reseal(p)
        plans.append(p)
    return plans


def fixture():
    arms = pd.DataFrame([dict(model_id=f"m_{s}_{seed}", source=s, seed=seed, checkpoint=f"checkpoint_{s}_{seed}", final_sha256=f"hash_{s}_{seed}")
        for s, seed in product(("cp_hk", "ukr_rus"), range(3))])
    cells, episodes, cohorts, audits = [], [], [], []
    for stream, target, arm in product(STREAMS, PANEL, arms.to_dict("records")):
        key = dict(stream=stream, target=target, model_id=arm["model_id"], source=arm["source"], seed=arm["seed"],
            checkpoint=arm["checkpoint"], weights_sha256=arm["final_sha256"])
        for condition, draw in [("draw", d) for d in range(8)]+[("probability_ensemble", -1)]:
            cells.append({**key, "queries": 256, "condition": condition, "draw": draw, "nll": .3, "accuracy": .5, "f1": .5, "roc_auc": .6})
        for ep, draw in product(range(128), range(8)):
            episodes.append({**key, "queries": 2, "episode": ep, "draw": draw, "query_nll": .3,
                "query_correct": 1, "prediction_changes_vs_original": 0})
        for cohort, n in (("all", 256), ("has_context", 128), ("no_context", 128)):
            cohorts.append({**key, "cohort": cohort, "queries": n, "always_correct": n//2, "always_wrong": n//2,
                "correctness_flips": 0, "mean_probability_variance": .2 if arm["source"] == "cp_hk" else .1})
        audits.append({**key, "baseline_batches_bit_exact": 32, "query_invariance_batches": 256, "direct_suffix_checks": 256,
            "maximum_direct_error": 0., "all_support_ids_and_labels_unchanged": True, "all_weights_unchanged": True})
    return (*[pd.DataFrame(v) for v in (cells, episodes, cohorts, audits)], arms)


class FixedContextAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = fixture()

    def test_complete_accounting(self):
        validate_frames(*self.frames)
        self.assertEqual(validate_plans(tiny_plans(), batches=1, draws=2), 1600)

    def test_support_identity_change_rejected_even_after_rehashing_plan(self):
        plans = tiny_plans()
        plans[0]["rows"][1]["support_centers"][0] = 999
        reseal(plans[0])
        with self.assertRaisesRegex(ValueError, "support identity"):
            validate_plans(plans, batches=1, draws=2)

    def test_missing_or_changed_draw_rejected(self):
        plans = tiny_plans()
        plans[0]["rows"][1]["seed"] += 1
        reseal(plans[0])
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_plans(plans, batches=1, draws=2)
        cells, ep, cohorts, audits, arms = self.frames
        with self.assertRaises(ValueError):
            validate_frames(cells.iloc[1:], ep, cohorts, audits, arms)

    def test_changed_query_path_rejected(self):
        cells, ep, cohorts, audits, arms = self.frames
        changed = audits.copy()
        changed.loc[0, "maximum_direct_error"] = 1e-8
        with self.assertRaisesRegex(ValueError, "invariants"):
            validate_frames(cells, ep, cohorts, changed, arms)

    def test_failed_primary_is_reported_not_filtered(self):
        cells, ep, cohorts, audits, arms = self.frames
        effects, contrast, result = summarize(cells, cohorts)
        self.assertTrue(result["primary_hong_kong_more_sensitive_every_seed_both_streams"])
        changed = cohorts.copy()
        mask = (changed.source == "cp_hk") & (changed.seed == 1) & (changed.stream == "fresh") & (changed.target == "covid_political")
        changed.loc[mask, "mean_probability_variance"] = .05
        _, contrast, result = summarize(cells, changed)
        self.assertFalse(result["primary_hong_kong_more_sensitive_every_seed_both_streams"])
        self.assertEqual(result["primary_positive_comparisons"], 5)
        self.assertEqual(len(contrast), 30)


if __name__ == "__main__":
    unittest.main()
