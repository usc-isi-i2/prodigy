import copy
import json
from pathlib import Path
import unittest

import pandas as pd

from .analyze_readout_interventions import MODES, READOUT_KEYS, UNCHANGED, METRICS, validate_manifest, validate_grid, parameter_effects
from .analyze_trajectories import PANEL, DECODERS


class ReadoutAnalysisTests(unittest.TestCase):
    def test_frozen_manifest_matches_complete_training_inventory(self):
        data = Path(__file__).parent / "data"
        payload = json.loads((data / "readout_interventions/manifest.json").read_text())
        receipt = json.loads((data / "readout_interventions/DONE.json").read_text())
        arms = pd.read_json(data / "member_training_verified/arms.json")
        initial = pd.read_csv(data / "member_initial_cells.csv")
        self.assertEqual(len(validate_manifest(payload, receipt, arms, initial)), 48)
        for key, value in (("terminal_sha256", "wrong-donor"), ("source", "other-source"),
                           ("changed_keys", ["layer_list.0.module_list.0.weight"])):
            bad = copy.deepcopy(payload)
            bad["models"][0][key] = value
            with self.assertRaises(ValueError):
                validate_manifest(bad, receipt, arms, initial)
        bad = copy.deepcopy(payload)
        bad["models"][-1] = bad["models"][0]
        with self.assertRaises(ValueError):
            validate_manifest(bad, receipt, arms, initial)

    def grid(self):
        manifests = []
        for mode, model_id, bg in zip(MODES, ("restore", "implant"), ("terminal", "initial")):
            manifests.append(dict(model_id=model_id, parent_model_id="terminal", source="ukr_rus", sources=["ukr_rus"],
                seed=0, policy="lowest_sorted", intervention=mode, checkpoint=f"/{model_id}.ckpt", weights_sha256=model_id,
                terminal_sha256="terminal", initial_sha256="initial", background_model_id=bg, changed_keys=sorted(READOUT_KEYS)))
        manifest = pd.DataFrame(manifests)
        cells, references, audit = [], [], []
        for target in PANEL:
            for model in manifest.itertuples():
                audit.append(dict(stream="original", dataset=target, target=target, model_id=model.model_id, episode_fingerprint=target))
                for decoder in DECODERS:
                    ref = dict(stream="original", dataset=target, decoder=decoder, model_id=model.background_model_id,
                        weights_sha256=model.background_model_id, episode_fingerprint=target, queries=3072,
                        roc_auc=.6, accuracy=.5, f1=.4, nll=.8)
                    references.append(ref)
                    value = {k: ref[k] for k in ("dataset", "decoder", "episode_fingerprint", "queries", *METRICS)}
                    value.update(model_id=model.model_id, weights_sha256=model.weights_sha256, checkpoint=model.checkpoint,
                                 sources=["ukr_rus"], variant="baseline", episodes=128)
                    if decoder not in UNCHANGED:
                        value["roc_auc"] += .1 if model.model_id == "restore" else -.2
                    cells.append(value)
        return pd.DataFrame(cells), manifest, pd.DataFrame(references), pd.DataFrame(audit).drop(columns="dataset")

    def test_grid_rejects_changed_control_metrics_or_wrong_weights(self):
        cells, manifest, references, audit = self.grid()
        valid = validate_grid(cells, manifest, references, audit, "original")
        self.assertEqual(len(valid), 170)
        for key, value in (("roc_auc", .7), ("weights_sha256", "different-hybrid"), ("nll", float("nan")),
                           ("episode_fingerprint", "different-inputs")):
            bad = cells.copy(deep=True)
            index = bad.index[bad.decoder == "S0_pool/ridge"][0]
            bad.loc[index, key] = value
            with self.assertRaises(ValueError):
                validate_grid(bad, manifest, references, audit, "original")
        with self.assertRaises(ValueError):
            validate_grid(cells.iloc[1:], manifest, references, audit, "original")

    def test_training_effect_signs_and_background_interaction(self):
        cells, manifest, references, audit = self.grid()
        changes = validate_grid(cells, manifest, references, audit, "original")
        effects = parameter_effects(changes)
        full = effects[effects.decoder == "full_model"]
        self.assertTrue(((full.trained_readout_effect_trained_background_roc_auc + .1).abs() < 1e-12).all())
        self.assertTrue(((full.trained_readout_effect_initial_background_roc_auc + .2).abs() < 1e-12).all())
        self.assertTrue(((full.background_by_readout_interaction_roc_auc - .1).abs() < 1e-12).all())
        with self.assertRaises(ValueError):
            parameter_effects(changes.iloc[1:])


if __name__ == "__main__":
    unittest.main()
