"""Exploratory exact-initialization reference, after the fixed primary analysis.

This is not a newly drawn random encoder and does not repair the missing step-zero
reference for the separate historical nine-source trajectory experiment.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze_trajectories import DECODERS, KEY, PANEL, read_jsonl
from .analyze_member_intervention import verify_receipt

METRICS = ("roc_auc", "accuracy", "f1", "nll")


def validate_manifest(manifest, arms):
    if len(manifest) != 3 or manifest.model_id.duplicated().any() or set(manifest.seed) != {0, 1, 2}:
        raise ValueError("exactly one initialization for each of three seeds required")
    if manifest.weights_sha256.nunique() != 3 or not all(s == [] for s in manifest.sources):
        raise ValueError("initializations must be distinct and have no trained sources")
    for row in manifest.itertuples():
        group = arms[arms.seed == row.seed]
        if len(group) != 8 or set(group.initial_sha256) != {row.weights_sha256}:
            raise ValueError("initial weights do not match all eight source/policy arms")
        possible = {str(Path(p).with_name("state_dict_0.ckpt")) for p in group.checkpoint}
        if row.checkpoint not in possible:
            raise ValueError("step-zero checkpoint is not a matched training-run artifact")


def validate_inputs(receipt, reference):
    expected = {(s, t) for s in ("original", "fresh") for t in PANEL}
    for item in (receipt, reference):
        if item.get("all_cached_batches_identical") is not True or item.get("stream_target_cells") != 10:
            raise ValueError("complete exact-input receipt required")
        rows = item.get("cells", [])
        if len(rows) != 10 or {(r["stream"], r["target"]) for r in rows} != expected:
            raise ValueError("missing or duplicate cached-input cell")
    lookup = {(r["stream"], r["target"]): r for r in reference["cells"]}
    for row in receipt["cells"]:
        prior = lookup[row["stream"], row["target"]]
        if len(row["batch_sha256"]) != 32 or any(row[k] != prior[k] for k in ("batch_sha256", "episode_fingerprint")):
            raise ValueError("initial and terminal replay inputs differ")


def validate_grid(cells, manifest, reference, stream):
    expected = pd.MultiIndex.from_product([PANEL, manifest.model_id, sorted(DECODERS)], names=KEY)
    if cells.empty or cells.duplicated(KEY).any() or set(pd.MultiIndex.from_frame(cells[KEY])) != set(expected):
        raise ValueError("incomplete initialization replay grid")
    if set(cells.variant) != {"baseline"} or set(cells.episodes) != {128} or not all(s == [] for s in cells.sources):
        raise ValueError("unexpected initialization replay protocol")
    if not np.isfinite(cells[list(METRICS)].to_numpy()).all():
        raise ValueError("nonfinite initialization metric")
    joined = cells.merge(manifest.drop(columns="sources"), on="model_id", suffixes=("", "_manifest"), validate="many_to_one")
    for key in ("checkpoint", "weights_sha256"):
        if not (joined[key] == joined[f"{key}_manifest"]).all():
            raise ValueError("initialization replay weights differ from verified starting weights")
    for target, group in joined.groupby("dataset"):
        prior = reference[reference.dataset == target]
        if group.episode_fingerprint.nunique() != 1 or set(group.episode_fingerprint) != set(prior.episode_fingerprint):
            raise ValueError("initialization episode fingerprint differs")
        if set(group.queries) != set(prior.queries):
            raise ValueError("initialization query count differs")
        for decoder, raw in group[group.decoder.str.startswith("raw_")].groupby("decoder"):
            ref = prior[prior.decoder == decoder]
            if ref.empty or any(raw[k].max() - raw[k].min() > 1e-8 or abs(raw[k].iloc[0] - ref[k].iloc[0]) > 1e-5 for k in METRICS):
                raise ValueError("raw-input reference changed")
    return joined.drop(columns=["checkpoint_manifest", "weights_sha256_manifest"]).assign(stream=stream)


def endpoint_changes(terminal, initial, arms):
    keys = ["stream", "dataset", "seed", "decoder"]
    if initial.duplicated(keys).any():
        raise ValueError("duplicate initial reference")
    cols = keys + ["model_id", "checkpoint", "weights_sha256", "episode_fingerprint"] + list(METRICS)
    joined = terminal.merge(initial[cols], on=keys, suffixes=("", "_initial"), validate="many_to_one")
    if len(joined) != len(terminal):
        raise ValueError("initial reference missing for a terminal cell")
    expected = joined.model_id.map(arms.set_index("model_id").initial_sha256)
    if not (joined.weights_sha256_initial == expected).all() or not (joined.episode_fingerprint == joined.episode_fingerprint_initial).all():
        raise ValueError("unmatched initialization or episode stream")
    for metric in METRICS:
        joined[f"change_0_to_2500_{metric}"] = joined[metric] - joined[f"{metric}_initial"]
    return joined


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[6] /
                        "scripts/experiments/setup/target_performance_mechanisms/data/member_initial_models.json")
    args = parser.parse_args()
    primary = json.loads((args.data / "member_intervention_validation.json").read_text())
    if primary.get("replay_rows") != 4080 or primary.get("models") != 24:
        raise ValueError("complete prespecified primary analysis required first")
    arms = pd.read_json(args.data / "member_training_verified/arms.json")
    verify_receipt(json.loads((args.data / "member_training_verified/DONE.json").read_text()), arms)
    manifest = pd.DataFrame(json.loads(args.manifest.read_text())["models"])
    validate_manifest(manifest, arms)
    validate_inputs(json.loads((args.data / "member_initial_input_validation.json").read_text()),
                    json.loads((args.data / "member_training_verified/input_validation.json").read_text()))
    terminal = pd.read_csv(args.data / "member_replay_cells.csv")
    if len(terminal) != 4080 or terminal.duplicated(["stream"] + KEY).any():
        raise ValueError("incomplete terminal reference")
    initial = pd.concat([validate_grid(read_jsonl(args.data / f"member_initial_{s}"), manifest,
                                      terminal[terminal.stream == s], s) for s in ("original", "fresh")], ignore_index=True)
    paired = initial[initial.stream == "original"].merge(initial[initial.stream == "fresh"], on=KEY, suffixes=("_original", "_fresh"))
    if len(paired) != 255 or (paired.episode_fingerprint_original == paired.episode_fingerprint_fresh).any():
        raise ValueError("both distinct episode streams required")
    initial.assign(sources=initial.sources.map(json.dumps)).to_csv(args.data / "member_initial_cells.csv", index=False)
    changes = endpoint_changes(terminal, initial, arms)
    changes.to_csv(args.data / "member_initial_to_terminal_changes.csv", index=False)
    summary = changes.groupby(["stream", "source", "policy", "dataset", "decoder"]).change_0_to_2500_roc_auc.agg(
        mean="mean", minimum="min", maximum="max", positive=lambda x: int((x > 0).sum()),
        negative=lambda x: int((x < 0).sum()), seeds="size")
    if not (summary.seeds == 3).all():
        raise ValueError("incomplete seed summary")
    summary.to_csv(args.data / "member_initial_to_terminal_summary.csv")
    (args.data / "member_initial_validation.json").write_text(json.dumps({
        "initialization_rows": len(initial), "terminal_changes": len(changes), "training_seeds": [0, 1, 2],
        "exact_initialization_shared_by_eight_models_per_seed": True, "all_320_cached_batch_hashes_match": True,
        "exploratory_supplement_added_after_training": True, "primary_analysis_completed_first": True,
        "additional_training": False, "checkpoint_selection": False, "historical_nine_source_step_zero_recovered": False,
    }, indent=2) + "\n")
    print(summary.loc[(slice(None), slice(None), "lowest_sorted", slice(None), ["S0_pool/ridge", "U1_pre_meta/ridge", "full_model"]), :].to_string())


if __name__ == "__main__":
    main()
