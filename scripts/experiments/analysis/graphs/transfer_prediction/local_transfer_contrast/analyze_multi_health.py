"""Evaluate support-conditioned transfer-health routing across all source experts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .analyze_contrast import (
    auc_inputs,
    fit_temperature,
    grouped_bootstrap_difference,
    softmax,
    to_numpy,
    validate_pair,
)


def auc(labels, scores):
    return float(roc_auc_score(labels, scores))


def learned_consensus(model):
    full = to_numpy(model["logits"]["full_model"]).argmax(1)
    names = [name for name in model["logits"] if not name.startswith("raw_") and name != "full_model"]
    votes = np.column_stack([to_numpy(model["logits"][name]).argmax(1) == full for name in names])
    return votes.mean(1) >= .5


def select_first_healthy(ranking, healthy):
    n = len(next(iter(healthy.values())))
    selected = np.full(n, ranking[0], dtype=object)
    unresolved = np.ones(n, dtype=bool)
    for model in ranking:
        choose = unresolved & healthy[model]
        selected[choose] = model
        unresolved[choose] = False
    return selected


def select_by_support_score(models, support_scores, healthy=None, tie_prior=None):
    """Per-query argmax with deterministic, label-free target-health tie breaking."""
    ranking = sorted(models, key=lambda name: (-(tie_prior or {}).get(name, 0.0), name))
    n = len(support_scores[ranking[0]])
    selected = np.full(n, ranking[0], dtype=object)
    best_score = np.full(n, -np.inf)
    any_eligible = np.zeros(n, dtype=bool)
    for model in ranking:
        eligible = np.ones(n, dtype=bool) if healthy is None else healthy[model]
        score = support_scores[model]
        choose = eligible & (score > best_score)
        selected[choose] = model
        best_score[choose] = score[choose]
        any_eligible |= eligible
    if healthy is not None and (~any_eligible).any():
        fallback = select_by_support_score(models, support_scores, tie_prior=tie_prior)
        selected[~any_eligible] = fallback[~any_eligible]
    return selected


def evaluate_selection(record, selected, temperatures, labels):
    logits = np.empty((len(selected), 2), dtype=np.float64)
    for name in np.unique(selected):
        mask = selected == name
        logits[mask] = to_numpy(record["models"][name]["logits"]["full_model"])[mask]
    scaled = np.stack([logit / temperatures[name] for logit, name in zip(logits, selected)])
    probabilities = softmax(scaled)
    predictions = logits.argmax(1)
    return predictions, probabilities


def analyze_target(path, target_name):
    discovery = torch.load(path / "original.pt", map_location="cpu", weights_only=False)
    validation = torch.load(path / "fresh.pt", map_location="cpu", weights_only=False)
    validate_pair(discovery, validation)
    y_discovery = to_numpy(discovery["labels"]["local_y"]).astype(int)
    y = to_numpy(validation["labels"]["local_y"]).astype(int)
    models = list(discovery["models"])
    temperatures = {
        model: fit_temperature(to_numpy(discovery["models"][model]["logits"]["full_model"]), y_discovery)
        for model in models
    }
    discovery_auc = {
        model: auc(*auc_inputs(discovery, softmax(to_numpy(
            discovery["models"][model]["logits"]["full_model"]))))
        for model in models
    }
    ranking = sorted(models, key=discovery_auc.get, reverse=True)
    best = ranking[0]
    raw_pred = to_numpy(validation["models"][best]["logits"]["raw_joint/ridge"]).argmax(1)
    health = {
        "u1_agreement": {
            model: to_numpy(validation["models"][model]["logits"]["full_model"]).argmax(1) ==
                   to_numpy(validation["models"][model]["logits"]["U1_pre_meta/ridge"]).argmax(1)
            for model in models
        },
        "raw_agreement": {
            model: to_numpy(validation["models"][model]["logits"]["full_model"]).argmax(1) == raw_pred
            for model in models
        },
        "learned_stage_consensus": {model: learned_consensus(validation["models"][model]) for model in models},
    }
    selections = {"fixed_best": np.full(len(y), best, dtype=object)}
    selections.update({name: select_first_healthy(ranking, values) for name, values in health.items()})
    if all("support_health" in validation["models"][model] for model in models):
        u1_rate = {model: float(health["u1_agreement"][model].mean()) for model in models}
        support_scores = {
            model: to_numpy(validation["models"][model]["support_health"]["u1_loo_prototype_accuracy"])
            for model in models
        }
        selections["support_loo"] = select_by_support_score(
            models, support_scores, tie_prior=u1_rate,
        )
        selections["support_loo_u1_agreement"] = select_by_support_score(
            models, support_scores, healthy=health["u1_agreement"], tie_prior=u1_rate,
        )
    # Query-label oracle is diagnostic only.
    correctness = {
        model: to_numpy(validation["models"][model]["logits"]["full_model"]).argmax(1) == y
        for model in models
    }
    oracle = np.full(len(y), best, dtype=object)
    for index in range(len(y)):
        oracle[index] = next((model for model in ranking if correctness[model][index]), best)
    selections["oracle_expert"] = oracle
    groups = to_numpy(validation["labels"]["episode_ids"])
    fixed_correct = correctness[best]
    rows = []
    for index, (method, selected) in enumerate(selections.items()):
        predictions, probabilities = evaluate_selection(validation, selected, temperatures, y)
        auc_y, auc_score = auc_inputs(validation, probabilities)
        correct = predictions == y
        low, high = grouped_bootstrap_difference(correct, fixed_correct, groups, 101 + index)
        counts = pd.Series(selected).value_counts()
        rows.append({
            "target": target_name, "method": method, "n": len(y), "best_model_from_discovery": best,
            "uses_validation_query_labels": method == "oracle_expert",
            "uses_any_query_labels_for_selection": method in {
                "fixed_best", "u1_agreement", "raw_agreement", "learned_stage_consensus", "oracle_expert"
            },
            "accuracy": float(accuracy_score(y, predictions)), "auc": auc(auc_y, auc_score),
            "nll": float(log_loss(y, probabilities, labels=[0, 1])),
            "accuracy_gain_over_fixed_best": float(correct.mean() - fixed_correct.mean()),
            "episode_bootstrap_gain_ci_low": low, "episode_bootstrap_gain_ci_high": high,
            "fraction_not_best": float(np.mean(selected != best)),
            "selection_counts": json.dumps(counts.to_dict(), sort_keys=True),
        })
    return rows, ranking, discovery_auc


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="target=export_dir")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, rankings = [], {}
    for item in args.inputs:
        target, raw_path = item.split("=", 1)
        target_rows, ranking, scores = analyze_target(Path(raw_path), target)
        rows.extend(target_rows)
        rankings[target] = {"ranking": ranking, "discovery_auc": scores}
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "multi_health_results.csv", index=False)
    non_oracle = frame[~frame.uses_validation_query_labels]
    macro = non_oracle.groupby("method").agg(
        targets=("target", "nunique"), mean_accuracy=("accuracy", "mean"),
        mean_auc=("auc", "mean"), mean_accuracy_gain=("accuracy_gain_over_fixed_best", "mean"),
        targets_improved=("accuracy_gain_over_fixed_best", lambda x: int((x > 0).sum())),
        targets_non_degraded=("accuracy_gain_over_fixed_best", lambda x: int((x >= 0).sum())),
    ).reset_index()
    macro.to_csv(args.output / "multi_health_macro.csv", index=False)
    (args.output / "rankings.json").write_text(json.dumps(rankings, indent=2, sort_keys=True) + "\n")
    print(frame.to_string(index=False))
    print("\nMacro\n", macro.to_string(index=False))


if __name__ == "__main__":
    main()
