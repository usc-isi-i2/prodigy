import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from .test_analyze_mixture_complementarity import fixture as prior_fixture
from .analyze_mixture_complementarity import validate_artifacts
from .analyze_mixture_budget import STEPS, TRAIN_FIELDS, budget_step, model_registry, validate_budget, summarize_budget, validate_training_audit


def fixture(root):
    inputs, _ = prior_fixture(root)
    prior = validate_artifacts(root, inputs)
    manifests, inventory = [], []
    singleton = prior["model_metrics"].query("source_count == 1").drop_duplicates("model_id")
    previous = prior["prediction_inventory"].drop_duplicates("model_id").set_index("model_id")
    for model in singleton.itertuples():
        for step in STEPS:
            name = model.model_id if step == 2500 else f"{model.model_id}__step{step}"
            checkpoint = previous.loc[model.model_id, "checkpoint"].replace("2500.ckpt", f"{step}.ckpt")
            manifests.append({"model_id": name, "checkpoint": checkpoint, "sources": model.sources[0]})
            inventory.append({"model_id": name, "checkpoint": checkpoint, "finite": True,
                "weights_sha256": hashlib.sha256(name.encode()).hexdigest()})
    models = model_registry(pd.DataFrame(manifests), pd.DataFrame(inventory), prior)
    tables = {name: [] for name in ("comparisons", "error_strata", "model_metrics", "prediction_inventory")}
    for name in ("comparisons", "error_strata"):
        for row in prior[name].to_dict("records"):
            for step in STEPS:
                k = row["source_count"]
                tables[name].append({**row, "specialist_step": step, "total_specialist_updates": k * step,
                    "specialist_training_episodes": 4 * k * step, "mixture_updates": 2500, "mixture_training_episodes": 10000,
                    "inference_model_multiplier": k, "budget_selected": step == budget_step(k)})
    for row in prior["model_metrics"].to_dict("records"):
        for step in STEPS if row["source_count"] == 1 else (2500,):
            name = row["model_id"] if step == 2500 else f"{row['model_id']}__step{step}"
            tables["model_metrics"].append({**row, "model_id": name, "step": step})
            old = prior["prediction_inventory"]
            old = old[old.stream.eq(row["stream"]) & old.target.eq(row["target"]) & old.model_id.eq(row["model_id"])].iloc[0].to_dict()
            m = models[models.model_id.eq(name)].iloc[0]
            tables["prediction_inventory"].append({**old, "model_id": name, "checkpoint": m.checkpoint, "weights_sha256": m.weights_sha256})
    tables = {name: pd.DataFrame(values) for name, values in tables.items()}
    tables["input_inventory"] = prior["input_inventory"].copy()
    return models, prior, tables


class MixtureBudgetTests(unittest.TestCase):
    def test_budget_requires_actual_configs_and_matching_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            models, _, _ = fixture(Path(tmp))
            audit = [{"model_id": m.model_id, "checkpoint": m.checkpoint, "config_sha256": "a" * 64,
                      "parameter_contract": {**TRAIN_FIELDS, "neighbor_sampling_source_subset": ",".join(m.sources),
                          "checkpoint_steps": "100,300,900,2500"}} for m in models[models.step.eq(2500)].itertuples()]
            validate_training_audit(audit, models)
            for key, value in (("batch_size", 32), ("neighbor_sampling_source_subset", "different"), ("checkpoint_steps", "100,2500")):
                changed = copy.deepcopy(audit)
                changed[0]["parameter_contract"][key] = value
                with self.assertRaises(ValueError):
                    validate_training_audit(changed, models)

    def test_budget_is_selected_by_updates_not_target(self):
        self.assertEqual((budget_step(2), budget_step(8)), (900, 300))
        with self.assertRaises(ValueError):
            budget_step(3)

    def test_all_steps_complete_and_negative_primary_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            models, prior, tables = fixture(Path(tmp))
            validate_budget(tables, models, prior)
            cells, summary = summarize_budget(tables["comparisons"])
            self.assertEqual((len(cells), len(summary)), (1800, 240))
            self.assertEqual(len(cells[cells.budget_selected]), 450)
            primary = cells[cells.target.eq("facebook_page_reference") & ~cells.target_seen & cells.source_count.eq(8) & cells.budget_selected]
            self.assertEqual(len(primary), 2)
            self.assertTrue(primary.ensemble_minus_mixture_auc.lt(0).all())

    def test_missing_step_wrong_budget_weights_and_stratum_arithmetic_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            models, prior, tables = fixture(Path(tmp))
            for table, column, value, message in (
                ("comparisons", "specialist_step", 42, "complete 1800"),
                ("comparisons", "total_specialist_updates", 1, "budget"),
                ("prediction_inventory", "weights_sha256", "a" * 64, "identity"),
                ("error_strata", "mixture_correct", 0, "conditional accuracy"),
                ("comparisons", "probability_ensemble_roc_auc", .2, "delta"),
            ):
                with self.subTest(table=table, column=column):
                    changed = copy.deepcopy(tables)
                    changed[table].loc[0, column] = value
                    with self.assertRaisesRegex(ValueError, message):
                        validate_budget(changed, models, prior)


if __name__ == "__main__":
    unittest.main()
