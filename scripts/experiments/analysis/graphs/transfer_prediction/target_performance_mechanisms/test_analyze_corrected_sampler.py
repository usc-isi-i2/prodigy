import copy
from itertools import product
import unittest

import pandas as pd

from .analyze_corrected_sampler import SOURCES, STEPS, PANEL, DECODERS, validate_inventory, validate_cells, summarize


def fixture():
    records = []
    for source, step in product(SOURCES, STEPS):
        records.append({"model_id": f"corrected_{source}_{step}", "source": source, "sources": [source], "step": step,
            "checkpoint": f"/{source}/state_dict_{step}.ckpt", "weights_sha256": f"new_{source}_{step}",
            "training_seed": 0, "completed_steps_verified": step, "optimizer_steps_verified": step,
            "effective_config": {"neighbor_sampling_source_subset": source, "seed": 0, "batch_size": 4, "epochs": 1,
                                 "dataset_len_cap": 2500, "task_name": "neighbor_matching"}})
    receipt = {"rows": 6120, "models": 36, "underlying_training_runs": 9, "training_seeds": [0], "steps": list(STEPS),
               "training_sidecars_and_configs_verified": True, "all_cached_inputs_match": True,
               "new_training": False, "matched_training_inputs_claim": False}
    manifest = {"models": records, "training_provenance": {"commit": "edd1649e8446416213d2e48893d641b3b697eca1"},
                "new_training": False, "changes_retention_and_roles_jointly": True, "matched_training_inputs_claim": False}
    corrected, old = [], []
    for stream, target, record, decoder in product(("original", "fresh"), PANEL, records, DECODERS):
        shared = {"stream": stream, "dataset": target, "decoder": decoder, "episodes": 128, "queries": 256,
                  "episode_fingerprint": stream + target, "variant": "baseline", "sources": record["sources"]}
        raw = decoder.startswith("raw_")
        new_auc = .6 if raw else .55 + record["step"] / 50000
        old_auc = .6 if raw else .55 + record["step"] / 25000
        corrected.append({**shared, **{k: record[k] for k in ("model_id", "checkpoint", "weights_sha256")},
            "roc_auc": new_auc, "accuracy": new_auc, "f1": new_auc, "nll": 1 - new_auc})
        old.append({**shared, "source": record["source"], "step": record["step"], "model_id": "old_" + record["model_id"],
            "checkpoint": "/old" + record["checkpoint"], "weights_sha256": "old_" + record["weights_sha256"],
            "roc_auc": old_auc, "accuracy": old_auc, "f1": old_auc, "nll": 1 - old_auc})
    return receipt, manifest, pd.DataFrame(corrected), pd.DataFrame(old)


class CorrectedAnalysisTests(unittest.TestCase):
    def test_full_grid_and_fixed_checkpoint_changes(self):
        receipt, manifest, cells, old = fixture()
        inventory = validate_inventory(receipt, manifest)
        joined, paired = validate_cells(cells, inventory, old)
        changes, summary, ranking = summarize(joined, paired, old)
        self.assertEqual((len(joined), len(paired), len(changes), len(summary), len(ranking)), (6120, 6120, 1530, 340, 340))
        row = changes[(changes.decoder == "full_model")].iloc[0]
        self.assertAlmostEqual(row.change_100_to_2500_corrected, .048)
        self.assertAlmostEqual(row.change_100_to_2500_historical, .096)
        self.assertAlmostEqual(row.difference_in_100_to_2500_changes, -.048)
        self.assertTrue(summary[summary.panel == "foreign"].sources.eq(8).all())
        self.assertTrue(ranking.descriptive_source_rank_correlation.isna().all())

    def test_inventory_requires_all_sources_steps_and_unmatched_design_disclosure(self):
        receipt, manifest, _, _ = fixture()
        for key, value in (("models", 9), ("matched_training_inputs_claim", True)):
            with self.assertRaises(ValueError):
                validate_inventory({**receipt, key: value}, manifest)
        changed = copy.deepcopy(manifest)
        changed["models"][0]["effective_config"]["neighbor_sampling_source_subset"] = "different"
        with self.assertRaisesRegex(ValueError, "training source"):
            validate_inventory(receipt, changed)
        with self.assertRaisesRegex(ValueError, "36 unique"):
            validate_inventory(receipt, {**manifest, "models": manifest["models"][:-1]})

    def test_missing_rows_weights_input_changes_and_raw_probe_drift_fail(self):
        receipt, manifest, cells, old = fixture()
        inventory = validate_inventory(receipt, manifest)
        with self.assertRaisesRegex(ValueError, "complete corrected"):
            validate_cells(cells.iloc[1:], inventory, old)
        for key, value, message in (("weights_sha256", "other", "checkpoint identity"),
                                    ("episode_fingerprint", "other", "query identities")):
            changed = cells.copy()
            changed.loc[0, key] = value
            with self.assertRaisesRegex(ValueError, message):
                validate_cells(changed, inventory, old)
        changed = cells.copy()
        changed.loc[changed.decoder.str.startswith("raw_"), "roc_auc"] += .1
        with self.assertRaisesRegex(ValueError, "raw input"):
            validate_cells(changed, inventory, old)


if __name__ == "__main__":
    unittest.main()
