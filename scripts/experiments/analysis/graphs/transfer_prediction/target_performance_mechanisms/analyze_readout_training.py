"""Prespecified frozen-readout training contrasts; complete grid or no result."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .analyze_member_initial_reference import METRICS, validate_inputs
from .analyze_member_intervention import validate_replay
from .analyze_trajectories import KEY, read_jsonl

SOURCES = {"ukr_rus", "cp_hk", "covid"}
READOUT_KEYS = {f"layer_list.0.reset_mlp_{p}.{k}" for p in ("c", "m") for k in ("weight", "bias")}


def validate_training(receipt, arms, inputs):
    flags = ("valid", "research_result", "exact_paired_inputs", "same_initialization", "same_final_walk_rng",
             "exact_frozen_tensors_all_updates_and_checkpoints", "complete_effective_config_checks")
    if not all(receipt.get(k) is True for k in flags) or any(receipt.get(k) != v for k, v in
            {"models": 18, "paired_comparisons": 9, "steps_per_model": 2500}.items()):
        raise ValueError("complete substantive training receipt required")
    expected = {(source, seed, condition) for source in SOURCES for seed in range(3) for condition in ("free", "frozen")}
    keys = ["source", "seed", "condition"]
    if len(arms) != 18 or arms.model_id.duplicated().any() or set(map(tuple, arms[keys].to_numpy())) != expected:
        raise ValueError("complete source/seed/condition grid required")
    if (arms.initial_sha256 == arms.final_sha256).any() or arms.initial_sha256.nunique() != 3 or not arms.groupby("seed").initial_sha256.nunique().eq(1).all():
        raise ValueError("missing matched distinct initialization or unchanged model")
    audit_inputs = pd.DataFrame(inputs)
    if len(audit_inputs) != 18 or audit_inputs.model_id.duplicated().any():
        raise ValueError("full training-input hash inventory incomplete")
    merged = arms.merge(audit_inputs, on=["model_id"] + keys, validate="one_to_one")
    if len(merged) != 18 or any(len(v) != 2500 for v in merged.input_hashes):
        raise ValueError("full training-input stream missing")
    for row in merged.itertuples():
        a = row.training_constraint
        if row.verified_checkpoints != [0, 100, 300, 900, 2500] or row.summary["steps"] != 2500 or row.summary["episodes"] != 10000:
            raise ValueError("wrong training budget or checkpoint audit")
        if a["condition"] != row.condition or a["initial_model_sha256"] != row.initial_sha256 or a["steps_checked"] != 2500 or a["inputs_hashed"] != 2500:
            raise ValueError("per-update constraint does not match completed run")
        names = sum(a["optimizer_parameter_names"], [])
        if set(a["readout_keys"]) != READOUT_KEYS or len(names) != len(set(names)) or (set(names) & READOUT_KEYS) != (READOUT_KEYS if row.condition == "free" else set()):
            raise ValueError("optimizer or frozen-parameter inventory differs")
    for _, group in merged.groupby(["source", "seed"]):
        rows = group.set_index("condition")
        for key in ("input_hashes", "initial_sha256", "final_walk_rng_sha256", "summary"):
            if rows.loc["free", key] != rows.loc["frozen", key]:
                raise ValueError(f"paired training mismatch: {key}")


def paired_changes(cells):
    keys = ["stream", "source", "seed", "dataset", "decoder"]
    free = cells[cells.condition == "free"]
    frozen = cells[cells.condition == "frozen"]
    paired = frozen.merge(free, on=keys, suffixes=("_frozen", "_free"), validate="one_to_one")
    if len(paired) != 1530 or len(free) != len(frozen) or not (paired.episode_fingerprint_frozen == paired.episode_fingerprint_free).all():
        raise ValueError("paired target-input grid incomplete or mismatched")
    for metric in METRICS:
        paired[f"delta_{metric}"] = paired[f"{metric}_frozen"] - paired[f"{metric}_free"]
    raw = paired[paired.decoder.str.startswith("raw_")]
    if (raw[[f"delta_{m}" for m in METRICS]].abs() > 1e-8).any().any():
        raise ValueError("raw input probes differ across training conditions")
    return paired


def primary_effect(changes):
    panel = changes[(changes.dataset == "facebook_page_reference") & (changes.decoder == "full_model")]
    expected = {(stream, source, seed) for stream in ("original", "fresh") for source in SOURCES for seed in range(3)}
    if len(panel) != 18 or set(map(tuple, panel[["stream", "source", "seed"]].to_numpy())) != expected:
        raise ValueError("primary source/seed grid incomplete")
    primary = panel.pivot(index=["stream", "seed"], columns="source", values="delta_roc_auc")
    primary["source_average_effect"] = primary[list(sorted(SOURCES))].mean(axis=1)
    per_source = panel.groupby(["stream", "source"]).delta_roc_auc.mean()
    supported = bool(primary.source_average_effect.gt(0).all() and per_source.gt(0).all())
    return primary.reset_index(), supported


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    root = args.data / "readout_training_verified"
    receipt = json.loads((root / "DONE.json").read_text())
    arms = pd.read_json(root / "arms.json")
    validate_training(receipt, arms, json.loads((root / "paired_input_hashes.json").read_text()))
    input_receipt = json.loads((root / "input_validation.json").read_text())
    validate_inputs(input_receipt, json.loads((args.data / "member_training_verified/input_validation.json").read_text()))
    reference = pd.read_csv(args.data / "member_replay_cells.csv")
    streams = []
    for stream in ("original", "fresh"):
        cells = validate_replay(read_jsonl(args.data / f"readout_training_{stream}"),
                                arms.assign(policy="lowest_sorted"), reference[reference.stream == stream], stream)
        for (target, decoder), raw in cells[cells.decoder.str.startswith("raw_")].groupby(["dataset", "decoder"]):
            prior = reference[(reference.stream == stream) & (reference.dataset == target) & (reference.decoder == decoder)]
            if prior.empty or any(raw[m].max() - raw[m].min() > 1e-8 or abs(raw[m].iloc[0] - prior[m].iloc[0]) > 1e-5 for m in METRICS):
                raise ValueError("raw input metrics differ from established reference")
        cells = cells.merge(arms[["model_id", "condition"]], on="model_id", validate="many_to_one")
        streams.append(cells)
    cells = pd.concat(streams, ignore_index=True)
    cross = streams[0].merge(streams[1], on=KEY, suffixes=("_original", "_fresh"), validate="one_to_one")
    if len(cross) != 1530 or not (cross.weights_sha256_original == cross.weights_sha256_fresh).all() or (cross.episode_fingerprint_original == cross.episode_fingerprint_fresh).any():
        raise ValueError("cross-stream weights or input independence differs")
    changes = paired_changes(cells)
    primary, supported = primary_effect(changes)
    summary = changes.groupby(["stream", "source", "dataset", "decoder"]).delta_roc_auc.agg(
        mean="mean", minimum="min", maximum="max", positive=lambda x: int((x > 0).sum()),
        negative=lambda x: int((x < 0).sum()), seeds="size").reset_index()
    if not summary.seeds.eq(3).all():
        raise ValueError("incomplete seed summary")
    # Reproduction is diagnostic; a new same-seed control is not another seed.
    old = reference[reference.policy == "lowest_sorted"]
    controls = cells[(cells.condition == "free") & cells.source.isin(["ukr_rus", "cp_hk"])]
    reproduction = controls.merge(old, on=["stream", "source", "seed", "dataset", "decoder"], suffixes=("", "_previous"), validate="one_to_one")
    if len(reproduction) != 1020 or not (reproduction.episode_fingerprint == reproduction.episode_fingerprint_previous).all():
        raise ValueError("previous standard-policy control comparison incomplete")
    for metric in METRICS:
        reproduction[f"delta_{metric}"] = reproduction[metric] - reproduction[f"{metric}_previous"]
    reproduction["same_weights_as_previous"] = reproduction.weights_sha256 == reproduction.weights_sha256_previous
    cells.assign(sources=cells.sources.map(json.dumps)).to_csv(args.data / "readout_training_cells.csv", index=False)
    changes.to_csv(args.data / "readout_training_changes.csv", index=False)
    primary.to_csv(args.data / "readout_training_primary.csv", index=False)
    summary.to_csv(args.data / "readout_training_summary.csv", index=False)
    reproduction.to_csv(args.data / "readout_training_control_reproduction.csv", index=False)
    (args.data / "readout_training_validation.json").write_text(json.dumps({
        "rows": len(cells), "paired_changes": len(changes), "models": 18, "training_seeds": [0, 1, 2],
        "full_model_cells": int((cells.decoder == "full_model").sum()), "training_receipt": receipt,
        "all_cached_inputs_match": True, "primary_consistently_positive": supported,
        "primary_target": "facebook_page_reference", "primary_decoder": "full_model",
        "exploratory_data_informed_followup": True, "untouched_domain_confirmation": False,
        "query_labels_fitted": False, "target_checkpoint_selection": False,
    }, indent=2) + "\n")
    print(primary.to_string(index=False))
    print(f"Prespecified positive-consistency criterion met: {supported}")


if __name__ == "__main__":
    main()
