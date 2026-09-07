import copy
import hashlib
from itertools import product
import json
import unittest

import pandas as pd

from .analyze_natural_support import validate_plan, validate_episode_accounting, summarize, METRICS, STREAMS


def hash_plan(p):
    p["sha256"] = hashlib.sha256(json.dumps({k: v for k, v in p.items() if k != "sha256"}, sort_keys=True).encode()).hexdigest()
    return p


def tiny_plan():
    audits = []
    for ep, local, draw in product(range(2), range(2), range(2)):
        global_class = local if ep == 0 else 1-local
        selected = 5-local if ep == 0 else 1-local
        audits.append(dict(batch=0, episode=ep, draw=draw, global_class=global_class, shots=1,
                           eligible_unique_ids=1, overlap_with_original=0, selected=[selected]))
    return hash_plan(dict(seed=13, draws=2, mappings=[[[5, 4, 2, 3, 1, 0, 6, 7]] * 2], audits=audits,
        pool_occurrences=4, pool_unique_identities=4, inputs=[dict(offset=0, episode_offset=0,
            tasks=[0]*4+[1]*4, centers=list(range(8)), query=[False, False, True, True]*2,
            support_labels=[0, 1, -1, -1]*2, classes=[[0, 1], [1, 0]])]))


class NaturalAnalysisTest(unittest.TestCase):
    def test_independent_plan_replay_and_query_identity_guards(self):
        plan = tiny_plan()
        self.assertEqual(validate_plan(plan, 2, 1, 1)["checked_support_positions"], 8)
        bad = copy.deepcopy(plan)
        bad["mappings"][0][0][2] = 5
        hash_plan(bad)
        with self.assertRaisesRegex(ValueError, "query identity"):
            validate_plan(bad, 2, 1, 1)
        bad = copy.deepcopy(plan)
        bad["inputs"][0]["support_labels"][2] = 0
        hash_plan(bad)
        with self.assertRaisesRegex(ValueError, "query-blind"):
            validate_plan(bad, 2, 1, 1)
        bad = copy.deepcopy(plan)
        bad["mappings"][0][0][0] = 0
        hash_plan(bad)
        with self.assertRaisesRegex(ValueError, "RNG/eligibility"):
            validate_plan(bad, 2, 1, 1)

    def test_episode_scores_reject_outcome_driven_selection(self):
        key = dict(stream="original", target="x", model_id="m")
        episodes = pd.DataFrame([key | dict(episode=e, draw=d, queries=1, support_cv_nll=float(d),
                                           query_nll=.3+.1*d, query_correct=int(d < 3), selected=d == 0)
                                 for e, d in product(range(128), range(8))])
        cells = pd.DataFrame([key | dict(condition="draw", draw=d, queries=128, accuracy=float(d < 3), nll=.3+.1*d) for d in range(8)]
                            + [key | dict(condition="support_cv_selected", draw=-1, queries=128, accuracy=1., nll=.3)])
        validate_episode_accounting(episodes, cells)
        episodes.loc[episodes.episode.eq(0), "selected"] = episodes.loc[episodes.episode.eq(0), "draw"].eq(1)
        with self.assertRaisesRegex(ValueError, "support selection"):
            validate_episode_accounting(episodes, cells)

    def test_failed_primary_cells_are_retained(self):
        cells, episodes, cohorts = [], [], []
        for stream, source, seed in product(STREAMS, ("cp_hk", "ukr_rus"), range(3)):
            key = dict(stream=stream, target="covid_political", model_id=f"{source}_{seed}", source=source, seed=seed)
            for condition in ("baseline", "support_cv_selected", "probability_ensemble"):
                value = .4 if condition == "support_cv_selected" else .5
                cells.append(key | dict(condition=condition, draw=-1, **{m: value for m in METRICS}))
            for draw in range(8):
                cells.append(key | dict(condition="draw", draw=draw, **{m: .5 for m in METRICS}))
                episodes.append(key | dict(episode=0, draw=draw, support_cv_nll=float(draw), query_nll=float(7-draw)))
            cohorts.append(key | dict(cohort="all", mean_probability_variance=.2 if source == "cp_hk" else .1))
        cells, episodes, cohorts = pd.DataFrame(cells), pd.DataFrame(episodes), pd.DataFrame(cohorts)
        _, relationships, _, result = summarize(cells, episodes, cohorts)
        self.assertTrue(result["hong_kong_more_sensitive_every_seed_both_streams"])
        self.assertTrue(result["support_cv_selection_lowers_hong_kong_nll_every_seed_both_streams"])
        self.assertTrue(relationships.within_episode_loss_correlation.between(-1.00000001, -.9999999).all())
        cells.loc[cells.source.eq("cp_hk") & cells.seed.eq(2) & cells.condition.eq("support_cv_selected"), "nll"] = .8
        cohorts.loc[cohorts.source.eq("cp_hk") & cohorts.seed.eq(1), "mean_probability_variance"] = .05
        _, _, _, result = summarize(cells, episodes, cohorts)
        self.assertFalse(result["hong_kong_more_sensitive_every_seed_both_streams"])
        self.assertFalse(result["support_cv_selection_lowers_hong_kong_nll_every_seed_both_streams"])


if __name__ == "__main__":
    unittest.main()
