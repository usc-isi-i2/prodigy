"""Validate and report the complete frozen-weight label-interface comparison."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze_trajectories import DECODERS, PANEL, read_jsonl

CONTROLS = {"zero_label_text", "train_label_table", "permuted_train_label_table",
            "train_norm_label_text", "zero_projected_labels"}
NEW_CONTROLS = CONTROLS - {"zero_label_text"}
KEY = ["dataset", "model_id", "variant", "decoder"]


def validate(cells, reference, stream):
    reference = reference[(reference.variant == "baseline") & (reference.model_id != "random_init")]
    models = sorted(reference.model_id.unique())
    if len(models) != 9:
        raise ValueError("requires all nine specialist references")
    expected = {(t, m, "baseline", d) for t in PANEL for m in models for d in DECODERS}
    expected |= {(t, m, v, "full_model") for t in PANEL for m in models for v in CONTROLS}
    if cells.duplicated(KEY).any() or set(map(tuple, cells[KEY].to_numpy())) != expected:
        raise ValueError("incomplete label-interface grid")
    if set(cells.episodes) != {128} or not np.isfinite(cells[["roc_auc", "accuracy", "f1", "nll"]].to_numpy()).all():
        raise ValueError("invalid budget or nonfinite metrics")
    identity = reference[reference.decoder == "full_model"]
    for _, row in cells.iterrows():
        r = identity[(identity.dataset == row.dataset) & (identity.model_id == row.model_id)].iloc[0]
        for field in ("weights_sha256", "checkpoint", "episode_fingerprint", "queries"):
            if row[field] != r[field]:
                raise ValueError(f"reference provenance mismatch: {field}")
        if row.variant in NEW_CONTROLS:
            a = row.label_interface_diagnostic
            if a["forward_calls"] != 33 or a["training_label_count"] != 120 or a["original_ignore_label_embeddings"] is not False:
                raise ValueError("label-interface audit incomplete")
            if not np.isfinite([a["last_projected_mean_norm"], a["last_interface_mean_norm"], a["training_table_mean_norm"]]).all():
                raise ValueError("nonfinite label norm")
            if row.variant == "train_norm_label_text" and abs(a["last_interface_mean_norm"] - a["training_table_mean_norm"]) > 1e-4:
                raise ValueError("label norm control failed")
            if row.variant == "zero_projected_labels" and a["last_interface_mean_norm"] != 0:
                raise ValueError("projected labels not zero")
            if row.variant == "permuted_train_label_table" and a["permutation_seed"] != 839113:
                raise ValueError("label permutation differs from plan")
    baseline = cells[cells.variant == "baseline"].merge(
        reference[KEY + ["roc_auc", "accuracy", "f1"]], on=KEY, validate="one_to_one", suffixes=("", "_reference"))
    if len(baseline) != 9 * len(PANEL) * len(DECODERS):
        raise ValueError("missing baseline reference")
    for metric, tolerance in (("roc_auc", 1e-5), ("accuracy", 1e-6), ("f1", 1e-6)):
        if ((baseline[metric] - baseline[f"{metric}_reference"]).abs() > tolerance).any():
            raise ValueError(f"baseline metric changed: {metric}")
    return cells.assign(stream=stream)


def contrasts(cells):
    full = cells[cells.decoder == "full_model"]
    baseline = full[full.variant == "baseline"][["stream", "dataset", "model_id", "roc_auc", "accuracy", "f1", "nll"]]
    deltas = full[full.variant != "baseline"].merge(baseline, on=["stream", "dataset", "model_id"],
                                                        suffixes=("", "_baseline"), validate="many_to_one")
    for metric in ("roc_auc", "accuracy", "f1", "nll"):
        deltas[f"delta_{metric}"] = deltas[metric] - deltas[f"{metric}_baseline"]
    return deltas


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    input_receipt = json.loads((args.data / "label_interface_input_validation.json").read_text())
    if input_receipt.get("all_cached_batches_identical") is not True or input_receipt.get("stream_target_cells") != 10:
        raise ValueError("complete exact cached-input receipt required")
    refs = {"original": pd.read_csv(args.data / "replay_cells.csv"),
            "fresh": pd.read_csv(args.data / "fresh_replay_cells.csv")}
    cells = pd.concat([validate(read_jsonl(args.data / f"label_interface_{s}"), ref, s) for s, ref in refs.items()], ignore_index=True)
    deltas = contrasts(cells)
    summary = deltas.groupby(["stream", "dataset", "variant"]).delta_roc_auc.agg(
        mean="mean", minimum="min", maximum="max", sources="size",
        positive=lambda x: int((x > 0).sum()), negative=lambda x: int((x < 0).sum())).reset_index()
    for name, table in (("cells", cells), ("deltas", deltas)):
        table.assign(sources=table.sources.map(json.dumps),
                     label_interface_diagnostic=table.label_interface_diagnostic.map(json.dumps)).to_csv(
                         args.data / f"label_interface_{name}.csv", index=False)
    summary.to_csv(args.data / "label_interface_summary.csv", index=False)
    (args.data / "label_interface_validation.json").write_text(json.dumps({
        "rows": len(cells), "full_model_cells": int((cells.decoder == "full_model").sum()),
        "models": 9, "training_seeds": [0], "same_cached_inputs": True,
        "same_weights_and_baseline_metrics": True, "complete_label_control_audits": True,
        "classification_label_inputs": "class-keyed deterministic standard-normal vectors, not semantic embeddings",
        "controls_fit_query_labels": False, "best_variant_selected": False,
    }, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
