import copy
from itertools import product
import unittest

import pandas as pd

from .analyze_role_context import interactions, source_gaps, summarize, validate_paired_record, VARIANTS, METRICS, STREAMS


class RoleAnalysisTest(unittest.TestCase):
    def test_error_accounting_and_class_conservation(self):
        counts = dict(queries=10, baseline_correct=6, altered_correct=7, errors_fixed=3, errors_introduced=2)
        row = counts | {"delta_accuracy": .1, "by_global_class": {"0": counts.copy()}}
        validate_paired_record(row)
        bad = copy.deepcopy(row)
        bad["by_global_class"]["0"]["errors_fixed"] = 2
        with self.assertRaisesRegex(ValueError, "sum"):
            validate_paired_record(bad)
        bad = copy.deepcopy(row)
        bad["errors_introduced"] = 0
        with self.assertRaisesRegex(ValueError, "accounting"):
            validate_paired_record(bad)

    def test_interaction_is_difference_of_differences(self):
        values = {"baseline": .5, "features_query": .6, "features_support": .65, "features_both": .8,
                  "edges_query": .4, "edges_support": .3, "edges_both": .2, "center_zero_query": .2}
        frame = pd.DataFrame([dict(stream="original", target="x", model_id="s", variant=v, **{m: value for m in METRICS}) for v, value in values.items()])
        result = interactions(frame).set_index("family")
        self.assertAlmostEqual(result.loc["features", "interaction_accuracy"], .05)
        self.assertAlmostEqual(result.loc["edges", "interaction_accuracy"], 0)

    def test_directional_expectation_requires_both_streams(self):
        rows = pd.DataFrame([dict(stream=s, target="covid_political", model_id="ss_ukr_rus", foreign_source=True,
                                 variant=v, **{"delta_" + m: (-.1 if v == "features_query" else 0.) for m in METRICS})
                             for s, v in product(STREAMS, VARIANTS)])
        _, result = summarize(rows)
        self.assertTrue(result["ukraine_political_query_context_dependence_both_streams"])
        rows.loc[(rows.stream == "fresh") & (rows.variant == "features_query"), "delta_roc_auc"] = .1
        _, result = summarize(rows)
        self.assertFalse(result["ukraine_political_query_context_dependence_both_streams"])

    def test_source_gap_decomposition_retains_negative_cohort_contribution(self):
        metrics, counts = [], []
        for stream, model in product(STREAMS, ("ss_ukr_rus", "ss_covid", "ss_cp_hk")):
            key = dict(stream=stream, target="covid_political", model_id=model)
            hk = model == "ss_cp_hk"
            for variant, auc in (("baseline", .6 if hk else .8), ("edges_support", .7 if hk else .8)):
                metrics.append(key | dict(variant=variant, roc_auc=auc))
            for cohort, n, correct in (("all", 20, 10 if hk else 15), ("no_context", 10, 6 if hk else 5),
                                       ("has_context", 10, 4 if hk else 10)):
                counts.append(key | dict(variant="baseline", cohort=cohort, queries=n, baseline_correct=correct))
        result = source_gaps(pd.DataFrame(metrics), pd.DataFrame(counts))
        self.assertEqual(len(result), 12)
        self.assertTrue(result[result.cohort.eq("no_context")].fraction_of_total_correct_gap.eq(-.2).all())
        self.assertTrue(result[result.cohort.eq("has_context")].fraction_of_total_correct_gap.eq(1.2).all())
        self.assertAlmostEqual(result.fraction_auc_gap_reduced.iloc[0], .5)


if __name__ == "__main__":
    unittest.main()
