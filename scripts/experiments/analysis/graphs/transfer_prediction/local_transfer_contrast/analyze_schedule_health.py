"""Test whether interleaving's transfer advantage coincides with higher U1 health."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch


PAIRS = {
    "cov_covpol": {
        "interleaved": "cov_covpol_interleaved",
        "sequential": ("cov_then_covpol", "covpol_then_cov"),
    },
    "cov_cphk": {
        "interleaved": "cov_cphk_interleaved",
        "sequential": ("cov_then_cphk", "cphk_then_cov"),
    },
}


def load_model_health(directory, model_id):
    records = torch.load(directory / f"{model_id}__baseline.pt", map_location="cpu", weights_only=False)
    if len(records) != 32:
        raise ValueError(f"incomplete replay for {directory.name}/{model_id}")
    full = torch.cat([record["logits"]["full_model"] for record in records]).argmax(1)
    u1 = torch.cat([record["logits"]["U1_pre_meta/ridge"] for record in records]).argmax(1)
    return float((full == u1).float().mean()), len(full)


def correlations(frame, scope):
    rows = []
    for metric in ("accuracy", "roc_auc"):
        result = spearmanr(frame.delta_u1_agreement, frame[f"delta_{metric}"])
        rows.append({"scope": scope, "metric": metric, "n": len(frame),
                     "spearman": float(result.statistic), "pvalue": float(result.pvalue),
                     "same_sign_fraction": float(np.mean(
                         np.sign(frame.delta_u1_agreement) == np.sign(frame[f"delta_{metric}"])
                     ))})
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a new output directory")
    if not (args.replay_root / "DONE").is_file():
        raise ValueError("complete replay required")
    args.output.mkdir(parents=True)
    model_to_pair = {}
    for pair, values in PAIRS.items():
        model_to_pair[values["interleaved"]] = (pair, "interleaved")
        model_to_pair.update({model: (pair, "sequential") for model in values["sequential"]})
    targets = sorted(path.name for path in args.replay_root.iterdir() if (path / "cache.json").is_file())
    if len(targets) != 5:
        raise ValueError("expected five target directories")
    rows = []
    for target in targets:
        directory = args.replay_root / target
        metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        full_metrics = {row["model_id"]: row for row in metrics
                        if row["variant"] == "baseline" and row["decoder"] == "full_model"}
        if set(full_metrics) != set(model_to_pair):
            raise ValueError(f"wrong model set for {target}")
        for model, metric in full_metrics.items():
            health, n = load_model_health(directory, model)
            pair, schedule = model_to_pair[model]
            rows.append({"target": target, "pair": pair, "model": model, "schedule": schedule,
                         "queries": n, "u1_agreement": health,
                         "accuracy": metric["accuracy"], "roc_auc": metric["roc_auc"],
                         "weights_sha256": metric["weights_sha256"],
                         "episode_fingerprint": metric["episode_fingerprint"]})
    health = pd.DataFrame(rows)
    comparisons = []
    for target in targets:
        for pair, values in PAIRS.items():
            part = health[(health.target == target) & (health.pair == pair)].set_index("model")
            baseline = part.loc[values["interleaved"]]
            for sequential in values["sequential"]:
                changed = part.loc[sequential]
                comparisons.append({
                    "target": target, "pair": pair, "sequential_model": sequential,
                    "interleaved_model": values["interleaved"], "queries": int(changed.queries),
                    "delta_u1_agreement": changed.u1_agreement - baseline.u1_agreement,
                    "delta_accuracy": changed.accuracy - baseline.accuracy,
                    "delta_roc_auc": changed.roc_auc - baseline.roc_auc,
                    "u1_agreement_interleaved": baseline.u1_agreement,
                    "u1_agreement_sequential": changed.u1_agreement,
                    "accuracy_interleaved": baseline.accuracy, "accuracy_sequential": changed.accuracy,
                    "roc_auc_interleaved": baseline.roc_auc, "roc_auc_sequential": changed.roc_auc,
                })
    comparisons = pd.DataFrame(comparisons)
    correlation_rows = correlations(comparisons, "all_targets")
    substantive = comparisons[~comparisons.target.eq("ukr_rus_suspended")]
    correlation_rows += correlations(substantive, "above_chance_targets")
    summary = comparisons.groupby("target").agg(
        comparisons=("sequential_model", "size"),
        mean_delta_u1_agreement=("delta_u1_agreement", "mean"),
        mean_delta_accuracy=("delta_accuracy", "mean"),
        mean_delta_roc_auc=("delta_roc_auc", "mean"),
        sequential_health_wins=("delta_u1_agreement", lambda x: int((x > 0).sum())),
        sequential_accuracy_wins=("delta_accuracy", lambda x: int((x > 0).sum())),
    ).reset_index()
    health.to_csv(args.output / "schedule_model_health.csv", index=False)
    comparisons.to_csv(args.output / "schedule_health_comparisons.csv", index=False)
    pd.DataFrame(correlation_rows).to_csv(args.output / "schedule_health_correlations.csv", index=False)
    summary.to_csv(args.output / "schedule_health_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\n", pd.DataFrame(correlation_rows).to_string(index=False))


if __name__ == "__main__":
    main()
