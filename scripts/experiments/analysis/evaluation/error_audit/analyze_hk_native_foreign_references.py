"""Compare HK and Ukraine class-reference components on identical canonical HK inputs."""
import argparse
import json
from pathlib import Path

import pandas as pd


COMPONENTS = ["positive", "negative", "self", "output_bias", "label_residual", "bn_offset"]


def summarize(frame, prefix):
    correct = frame[f"{prefix}_correct"]
    return {
        "n": len(frame),
        "accuracy": float(correct.mean()),
        "positive_gap_mean": float(frame[f"{prefix}_positive_gap"].mean()),
        "positive_gap_gt_zero": float(frame[f"{prefix}_positive_gap"].gt(0).mean()),
        "negative_gap_mean": float(frame[f"{prefix}_negative_gap"].mean()),
        "negative_gap_gt_zero": float(frame[f"{prefix}_negative_gap"].gt(0).mean()),
        "component_gap_means": {
            key: float(frame[f"{prefix}_{key}_gap"].mean()) for key in COMPONENTS
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hk", type=Path, required=True)
    parser.add_argument("--ukr", type=Path, required=True)
    parser.add_argument("--stages", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    keys = ["episode", "sample", "truth"]
    keep = keys + ["native_correct", "native_prediction", "positive_prediction", "support_prediction"] + [f"{x}_gap" for x in COMPONENTS]
    hk = pd.read_csv(args.hk)[keep].rename(columns={x: f"hk_{x}" for x in keep if x not in keys})
    ukr = pd.read_csv(args.ukr)[keep].rename(columns={x: f"ukr_{x}" for x in keep if x not in keys})
    data = hk.merge(ukr, on=keys, validate="one_to_one")
    data = data.rename(columns={"hk_native_correct": "hk_correct", "ukr_native_correct": "ukr_correct"})
    data["final_outcome"] = data.apply(
        lambda row: "both_correct" if row.hk_correct and row.ukr_correct else
        "hk_only" if row.hk_correct else "ukr_only" if row.ukr_correct else "both_wrong",
        axis=1,
    )

    stages = pd.read_csv(args.stages)
    stage_keys = ["episode", "sample"]
    stage = stages.pivot(index=stage_keys, columns="model", values=["encoded_correct", "native_correct"]).reset_index()
    stage.columns = ["_".join(str(x) for x in col if x) for col in stage.columns]
    assert len(stage) == len(data)
    data = data.merge(stage, on=stage_keys, validate="one_to_one")
    report = {
        "rows": len(data),
        "stage_accuracy": {
            "hk_pre_metagraph": float(data.encoded_correct_hk.mean()),
            "ukr_pre_metagraph": float(data.encoded_correct_ukr.mean()),
            "hk_final": float(data.native_correct_hk.mean()),
            "ukr_final": float(data.native_correct_ukr.mean()),
        },
        "final_outcome_counts": {str(k): int(v) for k, v in data.final_outcome.value_counts().items()},
        "by_final_outcome": {
            outcome: {"hk": summarize(frame, "hk"), "ukr": summarize(frame, "ukr")}
            for outcome, frame in data.groupby("final_outcome")
        },
        "prediction_agreement": float(data.hk_native_prediction.eq(data.ukr_native_prediction).mean()),
        "both_wrong_same_prediction": int(
            ((data.final_outcome == "both_wrong") & data.hk_native_prediction.eq(data.ukr_native_prediction)).sum()
        ),
        "validation": {
            "hk_accuracy_matches_stage": bool(data.hk_correct.eq(data.native_correct_hk).all()),
            "ukr_accuracy_matches_stage": bool(data.ukr_correct.eq(data.native_correct_ukr).all()),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
