"""Complete cross-task replay of a declared compatible NM-intervention subset."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze_trajectories import DECODERS, PANEL, KEY, read_jsonl


def validate(cells, manifest, inventory, reference, stream):
    expected = {(t, m, d) for t in PANEL for m in manifest.model_id for d in DECODERS}
    if len(manifest) != 30 or cells.duplicated(KEY).any() or set(map(tuple, cells[KEY].to_numpy())) != expected:
        raise ValueError("incomplete 30-checkpoint cross-task grid")
    if set(cells.variant) != {"baseline"} or set(cells.episodes) != {128}:
        raise ValueError("unexpected variant or episode budget")
    if not np.isfinite(cells[["roc_auc", "accuracy", "f1", "nll"]].to_numpy()).all():
        raise ValueError("nonfinite cross-task metric")
    joined = cells.merge(manifest, on="model_id", validate="many_to_one", suffixes=("", "_manifest"))
    if not (joined.checkpoint == joined.checkpoint_manifest).all() or not all(a == b for a, b in zip(joined.sources, joined.sources_manifest)):
        raise ValueError("cross-task checkpoint/source manifest mismatch")
    digests = inventory.set_index("model_id").weights_sha256
    if not (joined.weights_sha256 == joined.model_id.map(digests)).all():
        raise ValueError("cross-task weights differ from strict-load inventory")
    for target, group in joined.groupby("dataset"):
        ref = reference[(reference.dataset == target) & (reference.variant == "baseline")]
        if set(group.episode_fingerprint) != set(ref.episode_fingerprint) or set(group.queries) != set(ref.queries):
            raise ValueError("cross-task inputs differ from reference")
        for decoder, raw in group[group.decoder.str.startswith("raw_")].groupby("decoder"):
            known = ref[ref.decoder == decoder].roc_auc
            if known.empty or raw.roc_auc.max() - raw.roc_auc.min() > 1e-8 or abs(raw.roc_auc.iloc[0] - known.iloc[0]) > 1e-5:
                raise ValueError("raw input probe changed")
    return joined.drop(columns=["checkpoint_manifest", "sources_manifest", "forward_params", "classification_adaptation"]).assign(stream=stream)


def paired_deltas(cells):
    keys = ["stream", "checkpoint_role", "dataset", "decoder"]
    metrics = ["roc_auc", "accuracy", "f1", "nll"]
    baseline = cells[cells.original_model_id == "nmi_baseline_r8_s0"][keys + metrics]
    if baseline.duplicated(keys).any():
        raise ValueError("duplicate paired baseline")
    result = cells.merge(baseline, on=keys, suffixes=("", "_baseline"), validate="many_to_one")
    if len(result) != len(cells):
        raise ValueError("missing checkpoint-rule-specific baseline")
    for metric in metrics:
        result[f"delta_{metric}"] = result[metric] - result[f"{metric}_baseline"]
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--nm-results", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[6]
    manifest_path = repo / "scripts/experiments/setup/target_performance_mechanisms/data/campaign_cls_manifest_v2/manifest.json"
    protocol = json.loads(manifest_path.read_text())
    manifest = pd.DataFrame(protocol["models"])
    receipt = json.loads((args.data / "campaign_cls_checkpoint_inventory.json").read_text())
    inventory = pd.DataFrame(receipt["rows"])
    if receipt.get("manifest_sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest():
        raise ValueError("declared manifest differs from checkpoint compatibility audit")
    if len(inventory) != 30 or inventory.model_id.duplicated().any() or not inventory[["finite_weights", "strict_load", "finite_forward"]].all().all():
        raise ValueError("missing complete compatibility inventory")
    if set(inventory.model_id) != set(manifest.model_id) or not (inventory.checkpoint == inventory.model_id.map(manifest.set_index("model_id").checkpoint)).all():
        raise ValueError("checkpoint inventory differs from declared model paths")
    input_receipt = json.loads((args.data / "campaign_cls_input_validation.json").read_text())
    if input_receipt.get("all_cached_batches_identical") is not True or input_receipt.get("stream_target_cells") != 10:
        raise ValueError("exact cross-task input validation missing")
    refs = {"original": pd.read_csv(args.data / "replay_cells.csv"), "fresh": pd.read_csv(args.data / "fresh_replay_cells.csv")}
    cells = pd.concat([validate(read_jsonl(args.data / f"campaign_cls_{s}"), manifest, inventory, ref, s)
                       for s, ref in refs.items()], ignore_index=True)
    if (cells.groupby("model_id").weights_sha256.nunique() != 1).any():
        raise ValueError("cross-task weights changed between targets/streams")
    deltas = paired_deltas(cells)
    for name, table in (("cells", cells), ("deltas", deltas)):
        table.assign(sources=table.sources.map(json.dumps), flags=table["flags"].map(json.dumps)).to_csv(
            args.data / f"campaign_cls_{name}.csv", index=False)
    # NM comparisons belong ONLY to the originally selected checkpoint rule.
    nm = pd.read_csv(args.nm_results)
    selected = manifest[manifest.checkpoint_role == "source_val_selected"]
    original = nm[nm.model_id.isin(selected.original_model_id) & nm.target.isin(PANEL)].copy()
    if len(original) != 75 or original.duplicated(["model_id", "target"]).any():
        raise ValueError("incomplete selected-checkpoint NM reference panel")
    index = selected.set_index("original_model_id")
    hashes = inventory[inventory.model_id.isin(selected.model_id)].set_index("model_id").checkpoint_file_sha256
    if not (original.checkpoint_step == original.model_id.map(index.step)).all():
        raise ValueError("selected NM step mismatch")
    expected_hash = original.model_id.map(index.model_id).map(hashes)
    if not (original.checkpoint_sha256 == expected_hash).all():
        raise ValueError("selected NM file hash mismatch")
    base_nm = original[original.model_id == "nmi_baseline_r8_s0"].set_index("target").roc_auc
    original["delta_nm_auc"] = original.roc_auc - original.target.map(base_nm)
    original.to_csv(args.data / "campaign_original_selected_nm.csv", index=False)
    comparison = deltas[(deltas.checkpoint_role == "source_val_selected") & (deltas.decoder == "full_model")].merge(
        original[["model_id", "target", "delta_nm_auc"]].rename(columns={"model_id": "original_model_id", "target": "dataset"}),
        on=["original_model_id", "dataset"], validate="many_to_one")
    if len(comparison) != 150:
        raise ValueError("incomplete matched selected NM/CLS comparison")
    comparison.drop(columns=["sources", "flags"]).to_csv(args.data / "campaign_nm_cls_selected_deltas.csv", index=False)
    print("Common-6000-step TwiBot outcomes, each stream; one training seed:")
    print(deltas[(deltas.dataset == "twibot20") & (deltas.decoder == "full_model") & (deltas.checkpoint_role == "common6000")][
        ["stream", "original_model_id", "roc_auc", "delta_roc_auc"]].to_string(index=False))
    (args.data / "campaign_cls_validation.json").write_text(json.dumps({
        "rows": len(cells), "full_model_cells": int((cells.decoder == "full_model").sum()),
        "underlying_training_runs": 15, "training_seeds": [0], "checkpoint_rules": ["common6000", "source_val_selected"],
        "all_cached_inputs_match": True, "all_weights_match_inventory": True,
        "selected_nm_hashes_and_steps_match": True, "only_unseen_target_source": "twibot20",
        "excluded_models": protocol["excluded"], "classification_selected_methods_or_steps": False,
        "paired_checkpoint_rules_kept_separate": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
