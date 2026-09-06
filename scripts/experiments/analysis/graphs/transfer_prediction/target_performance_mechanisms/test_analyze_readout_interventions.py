import copy
import json
from pathlib import Path
import unittest

import pandas as pd

from .analyze_readout_interventions import MODES, READOUT_KEYS, UNCHANGED, METRICS, validate_manifest, validate_grid, parameter_effects, analyze_cues, baseline_cue_changes
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

    def test_complete_cue_grid_including_new_center_branch_and_constants(self):
        data = Path(__file__).parent / "data"
        arms = pd.read_json(data / "member_training_verified/arms.json")
        initial = pd.read_csv(data / "member_initial_cells.csv")
        references = pd.concat([pd.read_csv(data / "member_replay_cells.csv"), initial], ignore_index=True)
        manifest = validate_manifest(json.loads((data / "readout_interventions/manifest.json").read_text()),
                                     json.loads((data / "readout_interventions/DONE.json").read_text()), arms, initial)
        previous = pd.read_csv(data / "member_cue_alignment_cells.csv")
        keys = ["stream", "model_id", "decoder", "cue"]
        columns = [f"mean_within_episode_{k}" for k in ("spearman", "pearson", "decision_agreement")]
        prior = previous.set_index(keys)
        reference = references[references.dataset == "twibot20"].set_index(["stream", "model_id", "decoder"])
        decoders = ("S0_conv_center/ridge", "S0_pool/ridge", "U1_pre_meta/ridge", "full_model")
        cue_names = ("center_indegree", "raw_center", "raw_context")
        rows, base, cells = [], {}, []
        for stream in ("original", "fresh"):
            for model_id in references.model_id.unique():
                for decoder in decoders:
                    ref = reference.loc[stream, model_id, decoder]
                    for cue in cue_names:
                        key = (stream, model_id, decoder, cue)
                        row = dict(zip(keys, key)) | dict(weights_sha256=ref.weights_sha256,
                            episode_fingerprint=ref.episode_fingerprint, total_episodes=128, valid_episodes=128,
                            **{c: .2 for c in columns})
                        if key in prior.index:
                            row.update({c: prior.loc[key, c] for c in columns})
                        # Undefined correlations must survive every paired comparison.
                        if key == ("original", "memberinit_s0", "S0_conv_center/ridge", "center_indegree"):
                            row.update(valid_episodes=0, **{c: None for c in columns})
                        rows.append(row)
                        base[key] = row
            for model in manifest.itertuples():
                for decoder in decoders:
                    bg = reference.loc[stream, model.background_model_id, decoder]
                    cells.append(dict(stream=stream, dataset="twibot20", model_id=model.model_id, decoder=decoder,
                                      weights_sha256=model.weights_sha256, episode_fingerprint=bg.episode_fingerprint))
                    for cue in cue_names:
                        row = base[stream, model.background_model_id, decoder, cue].copy()
                        row.update(model_id=model.model_id, weights_sha256=model.weights_sha256)
                        if decoder not in UNCHANGED:
                            row.update(**{c: .1 for c in columns})
                        rows.append(row)
        cues = pd.DataFrame(rows)
        changes = analyze_cues(cues, manifest, pd.DataFrame(cells), references, previous)
        self.assertEqual(len(changes), 1152)
        constants = changes[(changes.background_model_id == "memberinit_s0") & (changes.stream == "original") &
                            (changes.decoder == "S0_conv_center/ridge") & (changes.cue == "center_indegree")]
        self.assertEqual(len(constants), 8)
        self.assertTrue(constants.delta_mean_within_episode_spearman.isna().all())
        self.assertEqual(len(baseline_cue_changes(cues, arms, initial)), 576)
        with self.assertRaises(ValueError):
            analyze_cues(cues.iloc[1:], manifest, pd.DataFrame(cells), references, previous)


if __name__ == "__main__":
    unittest.main()
