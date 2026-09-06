from itertools import product
import unittest

import pandas as pd

from scripts.experiments.setup.final_core.core_plan import SOURCES
from .analyze_episode_cardinality import summarize, TARGETS, STREAMS, VARIANTS, METRICS


def fixture():
    return pd.DataFrame([{"stream": stream, "target": target, "source": source,
        "model_id": "ss_" + source, "variant": variant, "foreign": source != target,
        **{"delta_" + k: (.02 if variant == "count_joint_attention" else .01) for k in METRICS}}
        for stream, target, source, variant in product(STREAMS, TARGETS, SOURCES, VARIANTS)])


class CardinalityAnalysisTest(unittest.TestCase):
    def test_both_targets_both_streams_required(self):
        frame = fixture()
        summary, result = summarize(frame)
        self.assertEqual(len(summary), 120)
        self.assertTrue(result["primary_prediction_passed"])
        frame.loc[(frame.target == "twibot20") & (frame.stream == "fresh") & (frame.variant == "count_joint_attention"), "delta_roc_auc"] = -.01
        _, result = summarize(frame)
        self.assertFalse(result["primary_prediction_passed"])
        self.assertEqual(sum(r["passed"] for r in result["primary_cells"]), 3)

    def test_positive_change_must_exceed_directional_control(self):
        frame = fixture()
        frame.loc[frame.variant == "count_inverse_attention", "delta_roc_auc"] = .03
        _, result = summarize(frame)
        self.assertFalse(result["primary_prediction_passed"])

    def test_target_seen_donor_cannot_create_primary_success(self):
        frame = fixture()
        frame.loc[frame.variant == "count_joint_attention", "delta_roc_auc"] = -.01
        frame.loc[(~frame.foreign) & (frame.variant == "count_joint_attention"), "delta_roc_auc"] = 1.
        _, result = summarize(frame)
        self.assertFalse(result["primary_prediction_passed"])
        self.assertTrue(all(r["positive_donors"] == 0 for r in result["primary_cells"]))


if __name__ == "__main__":
    unittest.main()
