import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from experiments.trainer import TrainerFS
from .analyze_mixture_predictions import evaluate_logits, ensemble_scores, error_strata, load_predictions, compare_predictions


def labels_for(y):
    return {"local_y": y, "mapping": torch.tensor([[0, 1]]).repeat(len(y), 1), "use_global": True}


class MixturePredictionTests(unittest.TestCase):
    def test_shared_comparison_preserves_metric_deltas_and_error_accounting(self):
        labels = labels_for(torch.tensor([0, 1, 0, 1]))
        a = torch.tensor([[2., 0.], [0., 2.], [0., 2.], [2., 0.]])
        b = torch.tensor([[2., 0.], [2., 0.], [0., 2.], [0., 2.]])
        members = torch.stack([a, b])
        member_scores = [evaluate_logits(value, labels) for value in members]
        row, strata = compare_predictions({"model_id": "m", "queries": 4}, members, member_scores, a, member_scores[0], labels)
        ensemble = ensemble_scores(members, labels)
        self.assertEqual(row["mixture_minus_probability_ensemble_roc_auc"], member_scores[0]["roc_auc"] - ensemble["probability"][0]["roc_auc"])
        self.assertEqual(sum(s["mixture_correct"] for s in strata) / 4, row["mixture_accuracy"])
        self.assertEqual([s["queries"] for s in strata], [1, 1, 2])

    def test_metrics_match_production_for_global_binary_and_facebook_local_labels(self):
        y = torch.tensor([0, 1, 0, 1])
        logits = torch.tensor([[2., 0.], [0., 3.], [.2, 1.], [2., .5]])
        ctx = SimpleNamespace(parameter={"task_name": "classification"}, is_feature_prediction=False, is_regression=False)
        for mapping, global_mode in ((torch.tensor([[0, 1], [1, 0]]), True),
                                     (torch.tensor([[4, 9], [12, 17]]), False)):
            tasks = torch.tensor([0, 0, 1, 1])
            graph = SimpleNamespace(task_label_map=mapping, task_id_per_sample=tasks)
            onehot = F.one_hot(y, 2).float()
            batch = [graph, None, onehot, None, None, torch.ones(8)]
            glob = TrainerFS._extract_global_classification_eval(ctx, batch, onehot, logits)
            expected = TrainerFS._compute_eval_metrics(ctx, onehot, logits, glob)
            actual = evaluate_logits(logits, {"local_y": y, "mapping": mapping[tasks], "use_global": global_mode})
            for key in expected:
                self.assertEqual(actual[key], expected[key])
            self.assertEqual(actual["nll"], F.cross_entropy(logits, onehot).item())

    def test_complementarity_partition_and_binary_unanimity(self):
        y = torch.tensor([0, 1, 0, 1])
        a = torch.tensor([[2., 0.], [0., 2.], [0., 2.], [2., 0.]])
        b = torch.tensor([[2., 0.], [2., 0.], [0., 2.], [0., 2.]])
        mixture = torch.tensor([[2., 0.], [0., 2.], [2., 0.], [0., 2.]])
        labels = labels_for(y)
        members = torch.stack([a, b])
        ensembles = ensemble_scores(members, labels)
        strata, extras = error_strata(members, mixture, ensembles, labels)
        self.assertEqual([r["queries"] for r in strata], [1, 1, 2])
        self.assertEqual(extras["mean_pairwise_disagreement"], .5)
        self.assertEqual(extras["constituent_query_oracle_accuracy"], .75)
        self.assertEqual(extras["mixture_fixes_all_wrong_fraction"], .25)
        self.assertEqual(extras["mixture_harms_all_correct_fraction"], 0)
        for rule in ensembles:
            torch.testing.assert_close(ensembles[rule][1], ensemble_scores(members.flip(0), labels)[rule][1], rtol=0, atol=0)
        wrong = copy.deepcopy(ensembles)
        wrong["probability"][1][2] = torch.tensor([1., 0.])
        with self.assertRaisesRegex(ValueError, "unanimous"):
            error_strata(members, mixture, wrong, labels)

    def test_empty_stratum_is_undefined_not_zero(self):
        y = torch.tensor([0, 1])
        a = torch.tensor([[2., 0.], [2., 0.]])
        members = torch.stack([a, a])
        labels = labels_for(y)
        strata, _ = error_strata(members, a, ensemble_scores(members, labels), labels)
        mixed = next(r for r in strata if r["stratum"] == "mixed_correctness")
        self.assertEqual(mixed["queries"], 0)
        self.assertIsNone(mixed["mixture_accuracy"])
        self.assertIsNone(mixed["probability_ensemble_accuracy"])

    def test_extreme_logits_have_stable_ensemble_nll(self):
        y = torch.tensor([0, 1])
        labels = labels_for(y)
        a = torch.tensor([[-1000., 1000.], [1000., -1000.]])
        result = ensemble_scores(torch.stack([a, a]), labels)
        self.assertEqual(result["probability"][0]["nll"], 2000.)
        self.assertEqual(result["logit"][0]["nll"], 2000.)

    def test_prediction_export_needs_metric_weight_and_input_parity(self):
        logits = torch.tensor([[2., 0.], [0., 2.]])
        labels = labels_for(torch.tensor([0, 1]).repeat(32))
        labels.update(target="twibot20", batch_counts=[2]*32, cache={"episode_fingerprint": "episode", "batch_sha256": [str(i) for i in range(32)]})
        model = {"model_id": "ss_x", "checkpoint": "checkpoint", "weights_sha256": "weights", "sources": ["x"]}
        metrics = evaluate_logits(logits.repeat(32, 1), labels)
        row = {**model, **metrics, "dataset": "twibot20", "variant": "baseline", "decoder": "full_model", "episode_fingerprint": "episode", "queries": 64}
        records = [{"batch": i, "batch_sha256": str(i), "logits": {"full_model": logits.clone()}} for i in range(32)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            torch.save(records, path / "ss_x__baseline.pt")
            value, result, _ = load_predictions(path, model, labels, [row])
            self.assertEqual(len(value), 64)
            self.assertEqual(result, metrics)
            for field, bad in (("weights_sha256", "other"), ("roc_auc", .4), ("dataset", "covid_political")):
                with self.assertRaises(ValueError):
                    load_predictions(path, model, labels, [{**row, field: bad}])
            records[-1]["batch_sha256"] = "changed features"
            torch.save(records, path / "ss_x__baseline.pt")
            with self.assertRaises(ValueError):
                load_predictions(path, model, labels, [row])


if __name__ == "__main__":
    unittest.main()
