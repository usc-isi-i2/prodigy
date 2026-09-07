"""Aggregate transfer-health results across independently trained checkpoints."""
import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


def parse_seed_path(value):
    raw_seed, raw_path = value.split("=", 1)
    return int(raw_seed), Path(raw_path)


def centered_correlation(frame, stream, metric):
    part = frame[frame.stream.eq(stream)].copy()
    groups = [part.target, part.seed]
    health = part.u1_agreement_rate - part.groupby(["target", "seed"]).u1_agreement_rate.transform("mean")
    quality = part[metric] - part.groupby(["target", "seed"])[metric].transform("mean")
    result = spearmanr(health, quality)
    return {"stream": stream, "scope": "target_seed_centered_pool", "metric": metric, "n": len(part),
            "spearman": float(result.statistic), "pvalue": float(result.pvalue)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--seed-results", nargs="+", required=True, help="seed=analysis_directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a new output directory")
    parsed = [parse_seed_path(value) for value in args.seed_results]
    if sorted(seed for seed, _ in parsed) != [0, 1, 2]:
        raise ValueError("exactly checkpoint seeds 0, 1, and 2 are required")
    args.output.mkdir(parents=True)

    multi, health, correlations = [], [], []
    for seed, path in parsed:
        for filename, destination in (
            ("multi_health_results.csv", multi),
            ("model_health.csv", health),
            ("health_correlations.csv", correlations),
        ):
            frame = pd.read_csv(path / filename)
            frame.insert(0, "seed", seed)
            destination.append(frame)
    multi = pd.concat(multi, ignore_index=True)
    health = pd.concat(health, ignore_index=True)
    correlations = pd.concat(correlations, ignore_index=True)
    multi.to_csv(args.output / "multi_health_all_seeds.csv", index=False)
    health.to_csv(args.output / "model_health_all_seeds.csv", index=False)

    non_oracle = multi[~multi.uses_validation_query_labels]
    seed_macro = non_oracle.groupby(["seed", "method"]).agg(
        targets=("target", "nunique"), mean_accuracy=("accuracy", "mean"), mean_auc=("auc", "mean"),
        mean_accuracy_gain=("accuracy_gain_over_fixed_best", "mean"),
        targets_improved=("accuracy_gain_over_fixed_best", lambda x: int((x > 0).sum())),
        targets_non_degraded=("accuracy_gain_over_fixed_best", lambda x: int((x >= 0).sum())),
    ).reset_index()
    seed_macro.to_csv(args.output / "multi_health_seed_macro.csv", index=False)
    replication = non_oracle.groupby(["target", "method"]).agg(
        seeds=("seed", "nunique"), mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"),
        mean_auc=("auc", "mean"), std_auc=("auc", "std"),
        mean_accuracy_gain=("accuracy_gain_over_fixed_best", "mean"),
        min_accuracy_gain=("accuracy_gain_over_fixed_best", "min"),
        max_accuracy_gain=("accuracy_gain_over_fixed_best", "max"),
        seeds_improved=("accuracy_gain_over_fixed_best", lambda x: int((x > 0).sum())),
        seeds_non_degraded=("accuracy_gain_over_fixed_best", lambda x: int((x >= 0).sum())),
    ).reset_index()
    replication.to_csv(args.output / "multi_health_replication.csv", index=False)

    pooled = pd.DataFrame([
        centered_correlation(health, stream, metric)
        for stream in sorted(health.stream.unique()) for metric in ("accuracy", "auc")
    ])
    correlations.to_csv(args.output / "health_correlations_by_seed.csv", index=False)
    pooled.to_csv(args.output / "health_correlations_pooled_seeds.csv", index=False)
    source_summary = health.groupby(["stream", "model"]).agg(
        checkpoints=("seed", "nunique"), mean_accuracy=("accuracy", "mean"), mean_auc=("auc", "mean"),
        mean_u1_agreement=("u1_agreement_rate", "mean"), std_u1_agreement=("u1_agreement_rate", "std"),
    ).reset_index()
    source_summary.to_csv(args.output / "health_source_summary_all_seeds.csv", index=False)
    print(seed_macro.to_string(index=False))
    print("\nU1 replication\n", replication[replication.method.eq("u1_agreement")].to_string(index=False))
    print("\nPooled correlations\n", pooled.to_string(index=False))


if __name__ == "__main__":
    main()
