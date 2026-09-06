import hashlib
from itertools import combinations
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.experiments.setup.final_core.core_plan import SOURCES
from .analyze_mixture_complementarity import METRICS, TARGETS, rank_association, summarize, validate_artifacts


def fixture(root):
    inputs = root / "inputs"
    inputs.mkdir()
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    models, reference, scores, comparisons, strata, inventory, cached = [], [], [], [], [], [], []
    for count in (1, 2, 8):
        for source in combinations(sorted(SOURCES), count):
            model = "model_" + "__".join(source)
            models.append({"model_id": model, "sources": ",".join(source), "checkpoint": f"/example/{model}/state_dict_2500.ckpt"})
    for stream in ("original", "fresh"):
        for target in sorted(TARGETS):
            fingerprint = digest(stream + target)
            cached.append({"stream": stream, "target": target, "episodes": 128, "queries": 256,
                "batch_sha256": [digest(f"{stream}{target}{i}") for i in range(32)], "episode_fingerprint": fingerprint,
                "graph_path": f"/example/{target}.pt", "production_metric_uses_global_binary_labels": target != "facebook_page_reference"})
            for model in models:
                source = model["sources"].split(",")
                count = len(source)
                common = {"stream": stream, "target": target, "model_id": model["model_id"], "sources": source, "source_count": count}
                metrics = {"roc_auc": .7 if count == 1 else .8, "accuracy": .625 if count == 1 else .75,
                           "f1": .625 if count == 1 else .75, "nll": .6 if count == 1 else .5}
                scores.append({**common, **metrics})
                inventory.append({"stream": stream, "target": target, "model_id": model["model_id"], "checkpoint": model["checkpoint"],
                    "weights_sha256": digest(model["model_id"]), "prediction_file_sha256": digest(model["model_id"] + stream + target),
                    "queries": 256, "episode_fingerprint": fingerprint, "metric_reproduction_max_abs_error": 0})
                if stream == "original":
                    reference.append({"model_id": model["model_id"], "dataset": target, "sources": repr(source), **metrics,
                        "checkpoint_step": 2500, "training_seed": 0, "evaluation_seed": 0, "eval_episode_seed_offset": 0,
                        "episodes": 128, "n_way": 2, "n_shot": 10, "n_query": 1, "queries": 256, "architecture": "prodigy",
                        "task": "classification", "episode_fingerprint": fingerprint})
                if count == 1:
                    continue
                common.update(target_seen=target in source)
                row = {**common, "queries": 256, "mean_pairwise_disagreement": .25 * count / (2 * (count - 1)),
                       "constituent_query_oracle_accuracy": .75, "mixture_fixes_all_wrong_fraction": .125,
                       "mixture_harms_all_correct_fraction": .0625, "mixed_correctness_fraction": .25}
                for metric in METRICS:
                    single = {"roc_auc": .7, "accuracy": .625, "f1": .625, "nll": .6}[metric]
                    for stat in ("mean", "min", "max"):
                        row[f"constituent_{stat}_{metric}"] = single
                    row[f"mixture_{metric}"] = metrics[metric]
                    for rule in ("probability", "logit"):
                        row[f"{rule}_ensemble_{metric}"] = single
                        row[f"mixture_minus_{rule}_ensemble_{metric}"] = metrics[metric] - single
                        row[f"mixture_{rule}_ensemble_decision_agreement"] = .8
                        row[f"mixture_{rule}_ensemble_probability_l1"] = .2
                comparisons.append(row)
                for name, queries, mc, ec in (("all_correct", 128, 112, 128), ("all_wrong", 64, 32, 0), ("mixed_correctness", 64, 48, 32)):
                    strata.append({**common, "stratum": name, "queries": queries, "fraction_of_queries": queries / 256,
                        "mixture_correct": mc, "mixture_accuracy": mc / queries, "probability_ensemble_correct": ec,
                        "probability_ensemble_accuracy": ec / queries, "logit_ensemble_correct": ec, "logit_ensemble_accuracy": ec / queries})
    pd.DataFrame(reference).to_csv(inputs / "classification_long.tsv", sep="\t", index=False)
    pd.DataFrame(models).to_csv(inputs / "model_list.tsv", sep="\t", index=False)
    manifest = {field: hashlib.sha256((inputs / name).read_bytes()).hexdigest() for name, field in
                (("classification_long.tsv", "snapshot_metrics_sha256"), ("model_list.tsv", "snapshot_models_sha256"))}
    (inputs / "manifest.json").write_text(json.dumps(manifest))
    artifacts = {"model_metrics": scores, "comparisons": comparisons, "error_strata": strata,
        "prediction_inventory": inventory, "input_inventory": cached,
        "DONE": {"models": 54, "verified_model_target_stream_cells": 540, "mixture_comparisons": 450, "error_strata": 1350,
                 "all_logits_reproduce_metrics": True, "all_specialist_mixture_inputs_identical": True, "complete_both_streams": True},
        "protocol": {"new_training": False, "ensemble_weights_fitted": False, "query_labels_used_for_classifier_fitting": False,
                     "causal_interference_claim": False, "training_seeds": [0], "ensemble_training_and_inference_cost_multiplier": [2, 8]}}
    for name, value in artifacts.items():
        (root / f"{name}.json").write_text(json.dumps(value))
    return inputs, artifacts


class MixtureAnalysisTests(unittest.TestCase):
    def test_full_grid_accounting_and_foreign_panels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, _ = fixture(root)
            tables = validate_artifacts(root, inputs)
            comparisons, summary, associations, matched = summarize(tables)
            self.assertEqual((len(comparisons), len(summary), len(associations), len(matched)), (450, 60, 20, 225))
            self.assertTrue(summary.query("panel == 'foreign' and source_count == 2").mixtures.eq(28).all())
            self.assertTrue(summary.query("panel == 'foreign' and source_count == 8").mixtures.eq(1).all())
            self.assertTrue(comparisons.shared_error_net_accuracy_contribution.eq(.0625).all())
            self.assertTrue(comparisons.mixed_error_net_accuracy_contribution.eq(.0625).all())
            self.assertTrue(associations.partial_rank_correlation.isna().all())
            self.assertTrue(associations.partial_undefined_reason.eq("rank-deficient nuisance design").all())

    def test_fail_closed_for_missing_cells_metadata_drift_and_wrong_arithmetic(self):
        changes = [
            ("comparisons", lambda rows: rows.pop(), "grid"),
            ("error_strata", lambda rows: rows.__setitem__(0, rows[1]), "grid"),
            ("prediction_inventory", lambda rows: rows[0].update(weights_sha256="a" * 64), "weights differ"),
            ("comparisons", lambda rows: rows[0].update(target_seen=not rows[0]["target_seen"]), "membership"),
            ("comparisons", lambda rows: rows[0].update(mixture_minus_probability_ensemble_roc_auc=.4), "delta"),
            ("error_strata", lambda rows: rows[0].update(mixture_correct=100), "conditional rate"),
            ("comparisons", lambda rows: rows[0].update(mean_pairwise_disagreement=.6), "disagreement"),
            ("model_metrics", lambda rows: rows[0].update(roc_auc=.1), "official metric parity"),
        ]
        for name, change, message in changes:
            with self.subTest(name=name, message=message), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                inputs, artifacts = fixture(root)
                change(artifacts[name])
                (root / f"{name}.json").write_text(json.dumps(artifacts[name]))
                with self.assertRaisesRegex(ValueError, message):
                    validate_artifacts(root, inputs)

    def test_rank_adjustment_matches_orthogonal_projection_and_keeps_undefined_cases(self):
        rng = np.random.default_rng(12039)
        frame = pd.DataFrame(rng.normal(size=(28, 4)), columns=["mean_pairwise_disagreement", "gain", "constituent_mean_roc_auc", "constituent_auc_range"])
        result = rank_association(frame, "gain")
        ranked = frame.rank().to_numpy()
        q, _ = np.linalg.qr(np.column_stack([np.ones(28), ranked[:, 2:]]))
        residual = ranked[:, :2] - q @ (q.T @ ranked[:, :2])
        expected = residual[:, 0] @ residual[:, 1] / np.linalg.norm(residual[:, 0]) / np.linalg.norm(residual[:, 1])
        self.assertAlmostEqual(result["partial_rank_correlation"], expected)
        self.assertFalse(result["independent_pair_inference"])
        frame["gain"] = frame.constituent_mean_roc_auc
        self.assertEqual(rank_association(frame, "gain")["partial_undefined_reason"], "zero residual rank variance")
        with self.assertRaisesRegex(ValueError, "complete 28"):
            rank_association(frame.iloc[:-1], "gain")


if __name__ == "__main__":
    unittest.main()
