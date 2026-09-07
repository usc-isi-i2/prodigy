"""Validate and summarize stage replay; never pool targets into one headline AUC."""
import argparse
import json
from pathlib import Path

import pandas as pd


def validate(frame, expected_targets):
    key = ["dataset", "model_id", "variant", "decoder"]
    if frame.duplicated(key).any():
        raise ValueError("duplicate replay cells")
    if set(frame.dataset) != set(expected_targets):
        raise ValueError("target coverage is incomplete")
    for target, part in frame.groupby("dataset"):
        if len(part) != 250 or part.model_id.nunique() != 10:
            raise ValueError(f"{target}: expected 10 models x 25 diagnostics; got {len(part)}")
        if part.episode_fingerprint.nunique() != 1 or set(part.episodes) != {128}:
            raise ValueError(f"{target}: mixed or incomplete episodes")
        baseline = part[(part.variant == "baseline") & (part.decoder == "full_model") & (part.model_id != "random_init")]
        tolerance = baseline.get("official_auc_parity_atol", pd.Series(1e-6, index=baseline.index)).fillna(1e-6)
        decision_error = baseline.get("official_decision_metric_max_abs_error", baseline.official_metric_max_abs_error).fillna(baseline.official_metric_max_abs_error)
        if (tolerance > 1e-5).any() or (decision_error > 1e-6).any() or baseline.official_metric_max_abs_error.isna().any() or (baseline.official_metric_max_abs_error > tolerance).any():
            raise ValueError(f"{target}: official score parity is unproven")
        for decoder, raw in part[part.decoder.str.startswith("raw_")].groupby("decoder"):
            if raw.roc_auc.max() - raw.roc_auc.min() > 1e-8:
                raise ValueError(f"{target}: raw input/probe drift across checkpoints")
        if part.roc_auc.isna().any() or not part.roc_auc.between(0, 1).all():
            raise ValueError("invalid AUC")
    return frame


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--raw", type=Path, default=Path(__file__).parent / "data/replay")
    p.add_argument("--targets", default="covid_political,facebook_page_reference,election2020,twibot20,ukr_rus_suspended")
    args = p.parse_args()
    names = args.targets.split(",")
    rows = [json.loads(line) for name in names for line in (args.raw / f"{name}.jsonl").read_text().splitlines()]
    frame = validate(pd.DataFrame(rows), names)
    for col in ["sources"]:
        frame[col] = frame[col].map(json.dumps)
    frame.to_csv(args.raw.parent / "replay_cells.csv", index=False)
    full = frame[frame.decoder == "full_model"]
    base = full[full.variant == "baseline"][["dataset", "model_id", "roc_auc", "accuracy", "nll"]]
    deltas = full.merge(base, on=["dataset", "model_id"], suffixes=("", "_baseline"), validate="many_to_one")
    for metric in ("roc_auc", "accuracy", "nll"):
        deltas[f"delta_{metric}"] = deltas[metric] - deltas[f"{metric}_baseline"]
    deltas.to_csv(args.raw.parent / "replay_intervention_deltas.csv", index=False)
    stages = frame[frame.variant == "baseline"].pivot(index=["dataset", "model_id"], columns="decoder", values="roc_auc")
    stages.to_csv(args.raw.parent / "replay_stage_auc.csv")
    donor = stages.loc[(slice(None), ["ss_ukr_rus", "ss_twibot20"]), :].reset_index()
    differences = donor[donor.model_id == "ss_ukr_rus"].set_index("dataset").drop(columns="model_id") - donor[donor.model_id == "ss_twibot20"].set_index("dataset").drop(columns="model_id")
    differences.to_csv(args.raw.parent / "replay_ukraine_minus_twibot_auc.csv")
    receipt = {"rows": len(frame), "targets": names, "official_parity_cells": int(frame.official_metric_max_abs_error.notna().sum()),
               "max_official_metric_error": float(frame.official_metric_max_abs_error.max()),
               "training_seed_replication": False, "bn_batch_is_transductive": True,
               "inference": "Stage probes and fixed-input perturbations locate sensitivities; they do not causally identify training-source properties."}
    (args.raw.parent / "replay_validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    print(stages[["raw_center/ridge", "raw_context/ridge", "S0_pool/ridge", "U1_pre_meta/ridge", "full_model"]].round(4).to_string())


if __name__ == "__main__":
    main()
