"""Test whether label-free layer agreement predicts source transfer quality."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import torch

from .analyze_contrast import auc_inputs, softmax, to_numpy, validate_pair


def model_rows(record, target, stream):
    y = to_numpy(record["labels"]["local_y"]).astype(int)
    rows = []
    for model, values in record["models"].items():
        full_logits = to_numpy(values["logits"]["full_model"])
        u1_logits = to_numpy(values["logits"]["U1_pre_meta/ridge"])
        raw_logits = to_numpy(values["logits"]["raw_joint/ridge"])
        full_pred, u1_pred, raw_pred = (item.argmax(1) for item in (full_logits, u1_logits, raw_logits))
        healthy = full_pred == u1_pred
        auc_y, auc_score = auc_inputs(record, softmax(full_logits))
        correct = full_pred == y
        rows.append({
            "target": target, "stream": stream, "model": model, "n": len(y),
            "accuracy": float(correct.mean()),
            "auc": float(roc_auc_score(auc_y, auc_score)),
            "u1_accuracy": float(np.mean(u1_pred == y)), "raw_accuracy": float(np.mean(raw_pred == y)),
            "u1_agreement_rate": float(healthy.mean()),
            "accuracy_when_healthy": float(correct[healthy].mean()) if healthy.any() else np.nan,
            "accuracy_when_unhealthy": float(correct[~healthy].mean()) if (~healthy).any() else np.nan,
            "error_rate_ratio_unhealthy_to_healthy": (
                float((1 - correct[~healthy].mean()) / (1 - correct[healthy].mean()))
                if healthy.any() and (~healthy).any() and correct[healthy].mean() < 1 else np.nan
            ),
        })
    return rows


def correlation_rows(frame):
    rows = []
    for stream in sorted(frame.stream.unique()):
        part = frame[frame.stream.eq(stream)].copy()
        for target in sorted(part.target.unique()):
            group = part[part.target.eq(target)]
            for metric in ("accuracy", "auc"):
                result = spearmanr(group.u1_agreement_rate, group[metric])
                rows.append({"stream": stream, "scope": target, "metric": metric, "n": len(group),
                             "spearman": float(result.statistic), "pvalue": float(result.pvalue)})
        for metric in ("accuracy", "auc"):
            centered_health = part.u1_agreement_rate - part.groupby("target").u1_agreement_rate.transform("mean")
            centered_metric = part[metric] - part.groupby("target")[metric].transform("mean")
            result = spearmanr(centered_health, centered_metric)
            rows.append({"stream": stream, "scope": "target_centered_pool", "metric": metric,
                         "n": len(part), "spearman": float(result.statistic), "pvalue": float(result.pvalue)})
    return rows


def selector_rows(frame):
    rows = []
    for target in sorted(frame.target.unique()):
        original = frame[(frame.target == target) & (frame.stream == "original")]
        fresh = frame[(frame.target == target) & (frame.stream == "fresh")].set_index("model")
        choices = {
            "discovery_auc": original.sort_values("auc", ascending=False).iloc[0].model,
            "fresh_unlabeled_u1_agreement": fresh.sort_values("u1_agreement_rate", ascending=False).index[0],
            "fresh_oracle_auc": fresh.sort_values("auc", ascending=False).index[0],
        }
        for method, model in choices.items():
            selected = fresh.loc[model]
            rows.append({"target": target, "selector": method, "model": model,
                         "fresh_accuracy": selected.accuracy, "fresh_auc": selected.auc,
                         "fresh_u1_agreement_rate": selected.u1_agreement_rate})
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="target=export_dir")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in args.inputs:
        target, raw_path = item.split("=", 1)
        path = Path(raw_path)
        original = torch.load(path / "original.pt", map_location="cpu", weights_only=False)
        fresh = torch.load(path / "fresh.pt", map_location="cpu", weights_only=False)
        validate_pair(original, fresh)
        rows.extend(model_rows(original, target, "original"))
        rows.extend(model_rows(fresh, target, "fresh"))
    frame = pd.DataFrame(rows)
    correlations = pd.DataFrame(correlation_rows(frame))
    selectors = pd.DataFrame(selector_rows(frame))
    source_summary = frame.groupby(["stream", "model"]).agg(
        mean_accuracy=("accuracy", "mean"), mean_auc=("auc", "mean"),
        mean_u1_agreement=("u1_agreement_rate", "mean"),
    ).reset_index()
    frame.to_csv(args.output / "model_health.csv", index=False)
    correlations.to_csv(args.output / "health_correlations.csv", index=False)
    selectors.to_csv(args.output / "health_source_selection.csv", index=False)
    source_summary.to_csv(args.output / "health_source_summary.csv", index=False)
    print(correlations.to_string(index=False))
    print("\nSelectors\n", selectors.to_string(index=False))


if __name__ == "__main__":
    main()
