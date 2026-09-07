from itertools import product
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.experiments.setup.final_core.core_plan import SOURCES
from .analyze_episode_cardinality import summarize, validate, TARGETS, STREAMS, VARIANTS, METRICS, ROLES


def fixture():
    return pd.DataFrame([{"stream": stream, "target": target, "source": source,
        "model_id": "ss_" + source, "variant": variant, "foreign": source != target,
        **{"delta_" + k: (.02 if variant == "count_joint_attention" else .01) for k in METRICS}}
        for stream, target, source, variant in product(STREAMS, TARGETS, SOURCES, VARIANTS)])


def artifact_fixture(root):
    h = lambda value: hashlib.sha256(value.encode()).hexdigest()
    metrics, inventory, inputs, attention = [], [], [], []
    for stream, target in product(STREAMS, TARGETS):
        inputs.append({"stream": stream, "target": target, "episode_fingerprint": h(stream + target),
            "batch_sha256": [h(stream + target + str(i)) for i in range(32)], "graph_path": target + ".pt", "queries": 256})
        for source in SOURCES:
            base = {"stream": stream, "target": target, "model_id": "ss_" + source,
                "checkpoint": source + "/state_dict_2500.ckpt", "weights_sha256": h(source)}
            inventory.append(base | {"baseline_suffix_bit_exact_batches": 32,
                "embedding_file_sha256": h("embedding"), "prediction_file_sha256": h("prediction")})
            for variant in VARIANTS:
                metrics.append(base | {"source": source, "variant": variant,
                    "episode_fingerprint": h(stream + target), "episodes": 128, "queries": 256,
                    **{k: .6 for k in METRICS}})
                audit = {"eval_ways": 2, "eval_shots": 10, "train_ways": 30, "train_shots": 3,
                    "label_nodes": 8, "support_nodes": 80, "query_nodes": 8,
                    "attention_mass_per_head": {role: [1 / len(roles)] * 8 for roles in ROLES.values() for role in roles}}
                attention.append({k: base[k] for k in ("stream", "target", "model_id")} | {"variant": variant, "batches": [audit] * 32})
    artifacts = {"metrics": metrics, "inventory": inventory, "input_inventory": inputs, "attention": attention,
        "protocol": {"variants": list(VARIANTS), "query_outcomes_used_for_intervention": False, "new_distinct_negative_classes": False},
        "DONE": {"full_model_cells": 630, "model_target_stream_cells": 90, "baseline_suffix_bit_exact_batches": 2880,
                 "all_weights_unchanged": True, "all_cached_inputs_verified": True, "all_baseline_metrics_reproduced": True}}
    for name, value in artifacts.items():
        (root / f"{name}.json").write_text(json.dumps(value))
    return artifacts


class CardinalityAnalysisTest(unittest.TestCase):
    def test_complete_grid_and_attention_conservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = artifact_fixture(root)
            cells, attention = validate(root)
            self.assertEqual((len(cells), len(attention)), (630, 630))
            artifacts["attention"][0]["batches"][0]["attention_mass_per_head"]["label_positive"][0] = 1
            (root / "attention.json").write_text(json.dumps(artifacts["attention"]))
            with self.assertRaisesRegex(ValueError, "conserve"):
                validate(root)

    def test_missing_cell_rejected_even_with_complete_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = artifact_fixture(root)
            (root / "metrics.json").write_text(json.dumps(artifacts["metrics"][:-1]))
            with self.assertRaisesRegex(ValueError, "630-cell"):
                validate(root)

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
