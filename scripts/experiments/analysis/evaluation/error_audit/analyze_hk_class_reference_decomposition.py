"""Stratify the private full-HK class-reference decomposition into publishable counts."""
import argparse
import json
from pathlib import Path

import pandas as pd


COMPONENTS = ["positive", "negative", "self", "output_bias", "label_residual", "bn_offset"]


def summarize(frame):
    n = len(frame)
    return {
        "n": n,
        "native_accuracy": float(frame.native_correct.mean()),
        "positive_gap_gt_zero": float(frame.positive_gap.gt(0).mean()),
        "negative_gap_gt_zero": float(frame.negative_gap.gt(0).mean()),
        "largest_rival_favoring_component": {
            str(k): {"count": int(v), "fraction": float(v / n)}
            for k, v in frame.largest_pro_error_component.value_counts().items()
        },
        "component_gap_means": {key: float(frame[f"{key}_gap"].mean()) for key in COMPONENTS},
        "positive_only_accuracy": float(frame.positive_prediction.eq(frame.truth).mean()),
        "positive_plus_negative_accuracy": float(frame.support_prediction.eq(frame.truth).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decomposition", type=Path, required=True)
    parser.add_argument("--overlap", type=Path, required=True)
    parser.add_argument("--failure-stages", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    decomposition = pd.read_csv(args.decomposition)
    overlap = pd.read_csv(args.overlap)
    stages = pd.read_csv(args.failure_stages)
    keys = ["episode", "sample"]
    assert len(decomposition) == len(overlap) == len(stages) == 61_440
    assert not any(frame.duplicated(keys).any() for frame in (decomposition, overlap, stages))
    assert decomposition[keys + ["truth"]].equals(overlap[keys + ["truth"]])
    data = decomposition.merge(
        overlap[keys + ["valid_test_candidates", "canonical_assigned", "canonical_multi"]],
        on=keys,
        validate="one_to_one",
    ).merge(
        stages[keys + ["query", "category", "query_frequency"]],
        on=keys,
        validate="one_to_one",
    )
    data["answerability"] = data.valid_test_candidates.eq(1).map({True: "unique", False: "ambiguous"})
    data["assigned_outcome"] = data.native_correct.map({True: "correct", False: "error"})
    data["multi_outcome"] = data.canonical_multi.map({True: "correct", False: "error"})
    per_query_accuracy = data.groupby("query").native_correct.mean()
    data["query_behavior"] = data["query"].map(
        lambda query: "always_wrong" if per_query_accuracy[query] == 0 else
        "always_correct" if per_query_accuracy[query] == 1 else "mixed"
    )
    error_rows = data[~data.native_correct].copy()
    error_nodes = error_rows.groupby("query").agg(
        occurrences=("query", "size"),
        positive_gap_mean=("positive_gap", "mean"),
        positive_favors_wrong_fraction=("positive_gap", lambda values: values.gt(0).mean()),
        negative_gap_mean=("negative_gap", "mean"),
    )
    episode_errors = error_rows.groupby("episode").agg(
        errors=("query", "size"),
        positive_favors_wrong_fraction=("positive_gap", lambda values: values.gt(0).mean()),
        positive_largest_fraction=("largest_pro_error_component", lambda values: values.eq("positive").mean()),
        positive_gap_mean=("positive_gap", "mean"),
    )

    report = {
        "rows": len(data),
        "all": summarize(data),
        "assigned_outcome": {key: summarize(value) for key, value in data.groupby("assigned_outcome")},
        "answerability_x_assigned_outcome": {
            f"{a}_{o}": summarize(value)
            for (a, o), value in data.groupby(["answerability", "assigned_outcome"])
        },
        "multi_outcome": {key: summarize(value) for key, value in data.groupby("multi_outcome")},
        "failure_stage": {
            key: summarize(value) for key, value in data[~data.native_correct].groupby("category")
        },
        "error_query_behavior": {
            key: summarize(value) for key, value in data[~data.native_correct].groupby("query_behavior")
        },
        "error_node_weighted": {
            "distinct_nodes": len(error_nodes),
            "mean_node_positive_favors_wrong_fraction": float(error_nodes.positive_favors_wrong_fraction.mean()),
            "nodes_positive_favors_wrong_majority": float(error_nodes.positive_favors_wrong_fraction.gt(0.5).mean()),
            "mean_node_positive_gap": float(error_nodes.positive_gap_mean.mean()),
            "mean_node_negative_gap": float(error_nodes.negative_gap_mean.mean()),
        },
        "episode_distribution": {
            "episodes": len(episode_errors),
            "positive_favors_wrong_fraction_min": float(episode_errors.positive_favors_wrong_fraction.min()),
            "positive_favors_wrong_fraction_median": float(episode_errors.positive_favors_wrong_fraction.median()),
            "positive_favors_wrong_fraction_max": float(episode_errors.positive_favors_wrong_fraction.max()),
            "positive_largest_fraction_min": float(episode_errors.positive_largest_fraction.min()),
            "positive_largest_fraction_median": float(episode_errors.positive_largest_fraction.median()),
            "positive_largest_fraction_max": float(episode_errors.positive_largest_fraction.max()),
        },
        "validation": {
            "key_rows": len(data),
            "native_correct_matches_overlap": bool(data.native_correct.eq(data.canonical_assigned).all()),
            "assigned_errors": int((~data.native_correct).sum()),
            "multi_errors": int((~data.canonical_multi).sum()),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
