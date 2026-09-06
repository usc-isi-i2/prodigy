"""Exact within-background readout effects, retaining every declared comparison."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze_trajectories import DECODERS, KEY, PANEL, read_jsonl
from .analyze_member_intervention import verify_receipt
from .analyze_member_initial_reference import validate_inputs

MODES = ("trained_background_initial_readout", "initial_background_trained_readout")
READOUT_KEYS = {f"layer_list.0.reset_mlp_{part}.{kind}" for part in ("c", "m") for kind in ("weight", "bias")}
UNCHANGED = {f"{s}/{m}" for s in ("raw_center", "raw_context", "raw_joint", "S0_conv_center", "S0_pool")
             for m in ("prototype", "ridge")}
METRICS = ("roc_auc", "accuracy", "f1", "nll")


def validate_manifest(payload, receipt, arms, initial):
    manifest = pd.DataFrame(payload["models"])
    if payload.get("no_training") is not True or set(payload.get("readout_keys", [])) != READOUT_KEYS:
        raise ValueError("unexpected readout intervention contract")
    if receipt.get("models") != 48 or not all(receipt.get(k) is True for k in (
            "no_training", "exact_donor_tensors_verified", "only_declared_readout_keys_changed")):
        raise ValueError("construction receipt missing")
    expected = {(parent, mode) for parent in arms.model_id for mode in MODES}
    if len(manifest) != 48 or manifest.model_id.duplicated().any() or set(zip(manifest.parent_model_id, manifest.intervention)) != expected:
        raise ValueError("incomplete or duplicate hybrid inventory")
    lookup = arms.set_index("model_id")
    initials = initial[["seed", "model_id", "weights_sha256"]].drop_duplicates()
    if len(initials) != 3 or initials.seed.duplicated().any():
        raise ValueError("ambiguous exact-initialization reference")
    initial_ids = initials.set_index("seed").model_id.to_dict()
    backgrounds = []
    for row in manifest.itertuples():
        parent = lookup.loc[row.parent_model_id]
        if any(getattr(row, k) != parent[k] for k in ("source", "seed", "policy")) or row.sources != [row.source]:
            raise ValueError("hybrid training provenance differs")
        if row.initial_sha256 != parent.initial_sha256 or row.terminal_sha256 != parent.final_sha256:
            raise ValueError("hybrid donor weights differ from verified training")
        if row.terminal_checkpoint != parent.checkpoint or row.initial_checkpoint != str(Path(parent.checkpoint).with_name("state_dict_0.ckpt")):
            raise ValueError("hybrid donor checkpoint path differs")
        if not row.all_other_tensors_identical or not set(row.changed_keys) or not set(row.changed_keys) <= READOUT_KEYS:
            raise ValueError("unexpected changed parameters")
        backgrounds.append(row.parent_model_id if row.intervention == MODES[0] else initial_ids[row.seed])
    return manifest.assign(background_model_id=backgrounds)


def validate_audit(audit, manifest):
    if audit.get("models") != 48 or audit.get("stream_target_model_cells") != 480 or audit.get("exact_donor_tensors_verified") is not True or audit.get("unchanged_upstream_prediction_tensors") != 153600:
        raise ValueError("complete persisted-weight and upstream-output audit required")
    weights = pd.DataFrame(audit["weights"])
    if len(weights) != 48 or weights.model_id.duplicated().any() or set(weights.model_id) != set(manifest.model_id):
        raise ValueError("incomplete saved-state audit")
    joined = weights.merge(manifest[["model_id", "weights_sha256", "changed_keys"]], on="model_id", suffixes=("", "_manifest"), validate="one_to_one")
    if not joined.exact_donor_tensors.all() or not (joined.weights_sha256 == joined.weights_sha256_manifest).all() or not all(set(a) == set(b) for a, b in zip(joined.changed_keys, joined.changed_keys_manifest)):
        raise ValueError("saved-state audit contradicts manifest")
    cells = pd.DataFrame(audit["cells"])
    keys = ["stream", "target", "model_id"]
    expected = {(s, t, m) for s in ("original", "fresh") for t in PANEL for m in manifest.model_id}
    if len(cells) != 480 or cells.duplicated(keys).any() or set(map(tuple, cells[keys].to_numpy())) != expected:
        raise ValueError("incomplete upstream-output cell grid")
    if not (cells.unchanged_prediction_tensors == 320).all():
        raise ValueError("not every upstream prediction was checked")
    lookup = manifest.set_index("model_id")
    for row in cells.itertuples():
        model = lookup.loc[row.model_id]
        ref_hash = model.terminal_sha256 if model.intervention == MODES[0] else model.initial_sha256
        if row.weights_sha256 != model.weights_sha256 or row.background_model_id != model.background_model_id or row.background_weights_sha256 != ref_hash:
            raise ValueError("upstream-output audit used a different hybrid or background")
    return cells


def validate_grid(cells, manifest, references, audit_cells, stream):
    expected = pd.MultiIndex.from_product([PANEL, manifest.model_id, sorted(DECODERS)], names=KEY)
    if cells.empty or cells.duplicated(KEY).any() or set(pd.MultiIndex.from_frame(cells[KEY])) != set(expected):
        raise ValueError("incomplete hybrid replay grid")
    if set(cells.variant) != {"baseline"} or set(cells.episodes) != {128} or not np.isfinite(cells[list(METRICS)].to_numpy()).all():
        raise ValueError("unexpected replay protocol or nonfinite metric")
    joined = cells.merge(manifest, on="model_id", suffixes=("", "_manifest"), validate="many_to_one")
    for key in ("checkpoint", "weights_sha256"):
        if not (joined[key] == joined[f"{key}_manifest"]).all():
            raise ValueError("replay did not use the declared hybrid state")
    if not all(a == b for a, b in zip(joined.sources, joined.sources_manifest)):
        raise ValueError("replay source provenance differs")
    joined = joined.drop(columns=["checkpoint_manifest", "weights_sha256_manifest", "sources_manifest"]).assign(stream=stream)
    ref_cols = ["stream", "dataset", "decoder", "model_id", "weights_sha256", "episode_fingerprint", "queries"] + list(METRICS)
    ref = references[ref_cols].rename(columns={"model_id": "background_model_id"})
    merged = joined.merge(ref, on=["stream", "dataset", "decoder", "background_model_id"], suffixes=("", "_background"), validate="many_to_one")
    if len(merged) != len(joined):
        raise ValueError("missing exact background reference")
    expected_hashes = np.where(merged.intervention == MODES[0], merged.terminal_sha256, merged.initial_sha256)
    if not (merged.weights_sha256_background == expected_hashes).all() or not (merged.episode_fingerprint == merged.episode_fingerprint_background).all() or not (merged.queries == merged.queries_background).all():
        raise ValueError("background checkpoint, episode, or query mismatch")
    audit = audit_cells[audit_cells.stream == stream].rename(columns={"target": "dataset"})
    checked = merged.merge(audit[["dataset", "model_id", "episode_fingerprint"]], on=["dataset", "model_id"], suffixes=("", "_audit"), validate="many_to_one")
    if len(checked) != len(merged) or not (checked.episode_fingerprint == checked.episode_fingerprint_audit).all():
        raise ValueError("evaluation metrics and exact-output audit disagree")
    for metric in METRICS:
        merged[f"delta_{metric}"] = merged[metric] - merged[f"{metric}_background"]
        if (merged.loc[merged.decoder.isin(UNCHANGED), f"delta_{metric}"].abs() > 1e-7).any():
            raise ValueError("unchanged upstream logits produced inconsistent metrics")
    return merged


def parameter_effects(cells):
    keys = ["stream", "source", "seed", "policy", "parent_model_id", "dataset", "decoder"]
    rows = []
    for group_keys, group in cells.groupby(keys):
        if len(group) != 2 or set(group.intervention) != set(MODES):
            raise ValueError("both swap directions required")
        row = dict(zip(keys, group_keys))
        for metric in METRICS:
            effects = group.set_index("intervention")[f"delta_{metric}"]
            trained = -effects[MODES[0]]  # T/T minus T/I: effect of trained readout at trained background.
            initial = effects[MODES[1]]   # I/T minus I/I: effect at initial background.
            row[f"trained_readout_effect_trained_background_{metric}"] = trained
            row[f"trained_readout_effect_initial_background_{metric}"] = initial
            row[f"background_by_readout_interaction_{metric}"] = trained - initial
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_cues(cues, manifest, cells, references, previous):
    keys = ["stream", "model_id", "decoder", "cue"]
    known_ids = set(manifest.model_id) | set(references.model_id)
    expected = {(s, m, d, c) for s in ("original", "fresh") for m in known_ids
                for d in ("S0_pool/ridge", "U1_pre_meta/ridge", "full_model")
                for c in ("center_indegree", "raw_center", "raw_context")}
    if len(cues) != 1350 or cues.duplicated(keys).any() or set(map(tuple, cues[keys].to_numpy())) != expected:
        raise ValueError("incomplete hybrid-plus-background cue grid")
    if set(cues.total_episodes) != {128} or not cues.valid_episodes.between(0, 128).all():
        raise ValueError("invalid cue episode count")
    columns = ["mean_within_episode_spearman", "mean_within_episode_pearson", "mean_within_episode_decision_agreement"]
    if not np.isfinite(cues.loc[cues.valid_episodes > 0, columns].to_numpy()).all() or not cues.loc[cues.valid_episodes == 0, columns].isna().all().all():
        raise ValueError("constant-cue handling or finite-correlation contract failed")
    provenance = ["stream", "model_id", "decoder", "weights_sha256", "episode_fingerprint"]
    known = pd.concat([cells[cells.dataset == "twibot20"][provenance], references[references.dataset == "twibot20"][provenance]]).drop_duplicates()
    checked = cues.merge(known, on=provenance[:3], suffixes=("", "_reference"), validate="many_to_one")
    if len(checked) != 1350 or any(not (checked[k] == checked[f"{k}_reference"]).all() for k in provenance[3:]):
        raise ValueError("cue provenance differs from validated primary results")
    prior = cues.merge(previous[keys + columns + ["valid_episodes"]], on=keys, suffixes=("", "_prior"), validate="one_to_one")
    if len(prior) != 432 or not (prior.valid_episodes == prior.valid_episodes_prior).all() or any(not np.allclose(prior[k], prior[f"{k}_prior"], atol=1e-9, rtol=0, equal_nan=True) for k in columns):
        raise ValueError("existing terminal cue results did not reproduce")
    treatment = cues.merge(manifest[["model_id", "source", "seed", "policy", "intervention", "background_model_id"]], on="model_id", validate="many_to_one")
    baseline = cues.rename(columns={"model_id": "background_model_id"})
    changes = treatment.merge(baseline, on=["stream", "background_model_id", "decoder", "cue"], suffixes=("", "_background"), validate="many_to_one")
    if len(changes) != 864:
        raise ValueError("missing cue background comparison")
    for column in columns:
        changes[f"delta_{column}"] = changes[column] - changes[f"{column}_background"]
    # No missing-value imputation and no implicit conversion into a mediation claim.
    return changes


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    data = args.data
    primary = json.loads((data / "member_intervention_validation.json").read_text())
    initial_validity = json.loads((data / "member_initial_validation.json").read_text())
    if primary.get("replay_rows") != 4080 or initial_validity.get("initialization_rows") != 510:
        raise ValueError("complete preceding primary and initialization analyses required")
    arms = pd.read_json(data / "member_training_verified/arms.json")
    verify_receipt(json.loads((data / "member_training_verified/DONE.json").read_text()), arms)
    initial = pd.read_csv(data / "member_initial_cells.csv")
    terminal = pd.read_csv(data / "member_replay_cells.csv")
    references = pd.concat([terminal, initial], ignore_index=True)
    manifest = validate_manifest(json.loads((data / "readout_interventions/manifest.json").read_text()),
                                 json.loads((data / "readout_interventions/DONE.json").read_text()), arms, initial)
    audit_root = data / "readout_audit"
    if not (audit_root / "DONE").is_file():
        raise ValueError("complete persisted-tensor and output audit required")
    audit = json.loads((audit_root / "verification.json").read_text())
    audit_cells = validate_audit(audit, manifest)
    validate_inputs(json.loads((audit_root / "input_validation.json").read_text()),
                    json.loads((data / "member_training_verified/input_validation.json").read_text()))
    streams = [validate_grid(read_jsonl(data / f"readout_replay_{s}"), manifest, references, audit_cells, s) for s in ("original", "fresh")]
    cells = pd.concat(streams, ignore_index=True)
    paired = streams[0].merge(streams[1], on=KEY, suffixes=("_original", "_fresh"), validate="one_to_one")
    if not (paired.weights_sha256_original == paired.weights_sha256_fresh).all() or (paired.episode_fingerprint_original == paired.episode_fingerprint_fresh).any():
        raise ValueError("cross-stream weights or input distinctness failed")
    cells.assign(sources=cells.sources.map(json.dumps), changed_keys=cells.changed_keys.map(json.dumps)).to_csv(data / "readout_intervention_cells.csv", index=False)
    effects = parameter_effects(cells)
    effects.to_csv(data / "readout_parameter_effects.csv", index=False)
    summary = cells.groupby(["stream", "intervention", "source", "policy", "dataset", "decoder"]).delta_roc_auc.agg(
        mean="mean", minimum="min", maximum="max", positive=lambda x: int((x > 0).sum()), negative=lambda x: int((x < 0).sum()), seeds="size")
    if not (summary.seeds == 3).all():
        raise ValueError("incomplete seed summary")
    summary.to_csv(data / "readout_intervention_summary.csv")
    cues = pd.read_json(audit_root / "cue_alignment.json")
    cue_changes = analyze_cues(cues, manifest, cells, references, pd.read_csv(data / "member_cue_alignment_cells.csv"))
    cues.to_csv(data / "readout_cue_cells.csv", index=False)
    cue_changes.to_csv(data / "readout_cue_changes.csv", index=False)
    (data / "readout_intervention_validation.json").write_text(json.dumps({"rows": len(cells), "full_model_cells": 480,
        "hybrid_models": 48, "training_seeds": [0, 1, 2], "upstream_tensor_comparisons": 153600,
        "all_initial_terminal_hybrid_digests_match": True, "all_cached_inputs_match": True,
        "cue_cells": len(cues), "cue_changes": len(cue_changes), "previous_terminal_cue_cells_reproduced": 432,
        "query_labels_fitted": False, "causal_mediation_claim": False, "causal_additive_stage_decomposition": False,
        "exploratory_after_member_primary": True}, indent=2) + "\n")
    print(summary.loc[(slice(None), slice(None), slice(None), "lowest_sorted", "twibot20", ["U1_pre_meta/ridge", "full_model"]), :].to_string())


if __name__ == "__main__":
    main()
