#!/usr/bin/env python3
"""Analyze schedule scaling, bounded replay, and TRACE target health."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import torch

from scripts.experiments.analysis.graphs.transfer_prediction.local_transfer_contrast.analyze_contrast import (
    auc_inputs,
    softmax,
    to_numpy,
    validate_pair,
)
from scripts.experiments.analysis.graphs.transfer_prediction.local_transfer_contrast.analyze_multi_health import (
    SUPPORT_COMPETENCE_THRESHOLD,
    select_by_support_score,
)
from scripts.experiments.setup.trace_schedule_scaling.make_plan import build_plan


TARGETS = (
    "election2020", "ukr_rus_suspended", "twibot20", "cp_hk",
    "facebook_page_reference",
)
SCHEDULES = ("blocked", "replay100", "interleaved")


def evaluate_logits(record, logits):
    y = to_numpy(record["labels"]["local_y"]).astype(int)
    values = to_numpy(logits).astype(np.float64)
    probabilities = softmax(values)
    auc_y, auc_score = auc_inputs(record, probabilities)
    return {
        "accuracy": float(accuracy_score(y, values.argmax(1))),
        "roc_auc": float(roc_auc_score(auc_y, auc_score)),
        "nll": float(log_loss(y, probabilities, labels=[0, 1])),
    }


def model_rows(record, target, stream):
    y = to_numpy(record["labels"]["local_y"]).astype(int)
    rows = []
    for model_id, model in record["models"].items():
        full = model["logits"]["full_model"]
        u1 = model["logits"]["U1_pre_meta/ridge"]
        full_pred = to_numpy(full).argmax(1)
        u1_pred = to_numpy(u1).argmax(1)
        metrics = evaluate_logits(record, full)
        support = to_numpy(
            model["support_health"]["u1_loo_prototype_accuracy"]
        )
        rows.append(
            {
                "target": target,
                "stream": stream,
                "model_id": model_id,
                "rung": int(model["rung"]),
                "seed": int(model["seed"]),
                "schedule": model["schedule"],
                "sources": ",".join(model["sources"]),
                "queries": len(y),
                "accuracy": metrics["accuracy"],
                "roc_auc": metrics["roc_auc"],
                "nll": metrics["nll"],
                "u1_accuracy": float(np.mean(u1_pred == y)),
                "u1_agreement": float(np.mean(full_pred == u1_pred)),
                "support_competence": float(support.mean()),
                "weights_sha256": model["weights_sha256"],
                "episode_fingerprint": record["receipts"]["episode_fingerprint"],
            }
        )
    return rows


def paired_contrasts(cells):
    rows = []
    keys = ["target", "stream", "rung", "seed"]
    for key, group in cells.groupby(keys):
        by_schedule = group.set_index("schedule")
        if set(by_schedule.index) != set(SCHEDULES):
            raise ValueError(f"incomplete schedule group {key}")
        reference = by_schedule.loc["interleaved"]
        for schedule in ("blocked", "replay100"):
            changed = by_schedule.loc[schedule]
            rows.append(
                dict(zip(keys, key))
                | {
                    "schedule": schedule,
                    "reference": "interleaved",
                    **{
                        f"delta_{metric}": float(changed[metric] - reference[metric])
                        for metric in (
                            "accuracy", "roc_auc", "nll", "u1_accuracy",
                            "u1_agreement", "support_competence",
                        )
                    },
                }
            )
    return pd.DataFrame(rows)


def crossed_interaction_bootstrap(frame, metric, rng_seed, draws=10000):
    targets = sorted(frame.target.unique())
    seeds = sorted(frame.seed.unique())
    lookup = frame.set_index(["target", "seed", "rung"])[metric]

    def effect(sampled_targets, sampled_seeds):
        rung2 = []
        larger = []
        for target in sampled_targets:
            for seed in sampled_seeds:
                rung2.append(lookup.loc[target, seed, 2])
                larger.extend((lookup.loc[target, seed, 3], lookup.loc[target, seed, 4]))
        return float(np.mean(rung2) - np.mean(larger))

    observed = effect(targets, seeds)
    rng = np.random.default_rng(rng_seed)
    samples = np.empty(draws)
    for index in range(draws):
        sampled_targets = rng.choice(targets, len(targets), replace=True)
        sampled_seeds = rng.choice(seeds, len(seeds), replace=True)
        samples[index] = effect(sampled_targets, sampled_seeds)
    return observed, np.quantile(samples, (0.025, 0.975))


def interaction_rows(contrasts):
    rows = []
    for stream in sorted(contrasts.stream.unique()):
        for schedule in ("blocked", "replay100"):
            part = contrasts[
                contrasts.stream.eq(stream) & contrasts.schedule.eq(schedule)
            ]
            for metric_index, metric in enumerate(("accuracy", "roc_auc")):
                column = f"delta_{metric}"
                value, interval = crossed_interaction_bootstrap(
                    part, column, 7100 + metric_index + 10 * (schedule == "replay100")
                )
                rows.append(
                    {
                        "stream": stream,
                        "schedule": schedule,
                        "reference": "interleaved",
                        "metric": metric,
                        "mean_delta_rung2": float(part[part.rung.eq(2)][column].mean()),
                        "mean_delta_rungs3_4": float(part[part.rung.gt(2)][column].mean()),
                        "rung2_minus_larger_interaction": value,
                        "crossed_seed_target_bootstrap_low": float(interval[0]),
                        "crossed_seed_target_bootstrap_high": float(interval[1]),
                    }
                )
    return pd.DataFrame(rows)


def health_correlations(contrasts):
    rows = []
    for stream in sorted(contrasts.stream.unique()):
        for scope, part in (
            ("all_targets", contrasts[contrasts.stream.eq(stream)]),
            (
                "above_chance_targets",
                contrasts[
                    contrasts.stream.eq(stream)
                    & ~contrasts.target.eq("ukr_rus_suspended")
                ],
            ),
        ):
            for metric in ("accuracy", "roc_auc"):
                x = part.delta_u1_agreement
                y = part[f"delta_{metric}"]
                result = spearmanr(x, y)
                rows.append(
                    {
                        "stream": stream,
                        "scope": scope,
                        "metric": metric,
                        "n": len(part),
                        "spearman": float(result.statistic),
                        "pvalue": float(result.pvalue),
                        "same_sign_fraction": float(np.mean(np.sign(x) == np.sign(y))),
                    }
                )
    return pd.DataFrame(rows)


def selected_logits(record, selected):
    logits = np.empty((len(selected), 2), dtype=np.float64)
    for model_id in np.unique(selected):
        mask = selected == model_id
        logits[mask] = to_numpy(record["models"][model_id]["logits"]["full_model"])[mask]
    return logits


def selector_rows(discovery, validation, target):
    rows = []
    for rung in (2, 3, 4):
        for seed in (0, 1, 2):
            model_ids = [
                model_id for model_id, model in validation["models"].items()
                if (model["rung"], model["seed"]) == (rung, seed)
            ]
            if len(model_ids) != 3:
                raise ValueError(f"incomplete selector group {target}/{rung}/{seed}")
            discovery_auc = {
                model_id: evaluate_logits(
                    discovery, discovery["models"][model_id]["logits"]["full_model"]
                )["roc_auc"]
                for model_id in model_ids
            }
            best = max(model_ids, key=discovery_auc.get)
            support_scores = {
                model_id: to_numpy(
                    validation["models"][model_id]["support_health"]
                    ["u1_loo_prototype_accuracy"]
                )
                for model_id in model_ids
            }
            agreement = {
                model_id: (
                    to_numpy(validation["models"][model_id]["logits"]["full_model"]).argmax(1)
                    == to_numpy(validation["models"][model_id]["logits"]["U1_pre_meta/ridge"]).argmax(1)
                )
                for model_id in model_ids
            }
            tie_prior = {
                model_id: float(values.mean()) for model_id, values in agreement.items()
            }
            selections = {
                "discovery_auc_fixed": np.full(
                    len(next(iter(support_scores.values()))), best, dtype=object
                ),
                "support_loo": select_by_support_score(
                    model_ids, support_scores, tie_prior=tie_prior
                ),
                "trace_support_loo_u1": select_by_support_score(
                    model_ids, support_scores, healthy=agreement, tie_prior=tie_prior
                ),
            }
            target_competence = max(
                float(values.mean()) for values in support_scores.values()
            )
            for method, selected in selections.items():
                metrics = evaluate_logits(validation, selected_logits(validation, selected))
                rows.append(
                    {
                        "target": target,
                        "rung": rung,
                        "seed": seed,
                        "method": method,
                        "uses_target_query_labels": method == "discovery_auc_fixed",
                        "target_support_competence": target_competence,
                        "target_supported_at_0_55": (
                            target_competence >= SUPPORT_COMPETENCE_THRESHOLD
                        ),
                        "fraction_not_discovery_best": float(np.mean(selected != best)),
                        "discovery_best_model": best,
                        **metrics,
                    }
                )
    return rows


def replay_vs_conventional(cells):
    original = cells[cells.stream.eq("original")]
    fresh = cells[cells.stream.eq("fresh")]
    rows = []
    for rung in (2, 3, 4):
        discovery = original[
            original.rung.eq(rung)
            & original.schedule.isin(("blocked", "interleaved"))
        ]
        schedule_auc = discovery.groupby("schedule").roc_auc.mean()
        baseline = str(schedule_auc.idxmax())
        for (target, seed), group in fresh[fresh.rung.eq(rung)].groupby(
            ["target", "seed"]
        ):
            by_schedule = group.set_index("schedule")
            replay = by_schedule.loc["replay100"]
            reference = by_schedule.loc[baseline]
            rows.append(
                {
                    "rung": rung,
                    "target": target,
                    "seed": seed,
                    "discovery_selected_conventional": baseline,
                    "delta_accuracy": float(replay.accuracy - reference.accuracy),
                    "delta_roc_auc": float(replay.roc_auc - reference.roc_auc),
                    "delta_u1_agreement": float(
                        replay.u1_agreement - reference.u1_agreement
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a new output directory")
    arms = build_plan()
    expected_models = {arm.model_id for arm in arms}
    args.output.mkdir(parents=True)
    cells = []
    selections = []
    for target in TARGETS:
        discovery = torch.load(
            args.input / target / "original.pt", map_location="cpu", weights_only=False
        )
        validation = torch.load(
            args.input / target / "fresh.pt", map_location="cpu", weights_only=False
        )
        validate_pair(discovery, validation)
        if set(discovery["models"]) != expected_models:
            raise ValueError(f"wrong model set for {target}")
        cells.extend(model_rows(discovery, target, "original"))
        cells.extend(model_rows(validation, target, "fresh"))
        selections.extend(selector_rows(discovery, validation, target))
    cells = pd.DataFrame(cells)
    if len(cells) != 270 or cells.weights_sha256.nunique() != 27:
        raise ValueError("expected 27 distinct models x five targets x two streams")
    contrasts = paired_contrasts(cells)
    interactions = interaction_rows(contrasts)
    correlations = health_correlations(contrasts)
    selectors = pd.DataFrame(selections)
    replay = replay_vs_conventional(cells)
    summary = cells.groupby(["stream", "rung", "schedule"]).agg(
        cells=("model_id", "size"),
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        mean_roc_auc=("roc_auc", "mean"),
        std_roc_auc=("roc_auc", "std"),
        mean_u1_agreement=("u1_agreement", "mean"),
        mean_support_competence=("support_competence", "mean"),
    ).reset_index()
    selector_summary = selectors.groupby("method").agg(
        cells=("target", "size"),
        mean_accuracy=("accuracy", "mean"),
        mean_roc_auc=("roc_auc", "mean"),
        supported_cells=("target_supported_at_0_55", "sum"),
    ).reset_index()
    replay_summary = replay.groupby("rung").agg(
        cells=("target", "size"),
        baseline=("discovery_selected_conventional", "first"),
        mean_delta_accuracy=("delta_accuracy", "mean"),
        mean_delta_roc_auc=("delta_roc_auc", "mean"),
        mean_delta_u1_agreement=("delta_u1_agreement", "mean"),
        accuracy_wins=("delta_accuracy", lambda values: int((values > 0).sum())),
        auc_wins=("delta_roc_auc", lambda values: int((values > 0).sum())),
    ).reset_index()
    for filename, frame in (
        ("cells.csv", cells),
        ("paired_contrasts.csv", contrasts),
        ("schedule_summary.csv", summary),
        ("scaling_interactions.csv", interactions),
        ("health_correlations.csv", correlations),
        ("selector_results.csv", selectors),
        ("selector_summary.csv", selector_summary),
        ("replay_vs_conventional.csv", replay),
        ("replay_vs_conventional_summary.csv", replay_summary),
    ):
        frame.to_csv(args.output / filename, index=False)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "models": 27,
                "targets": list(TARGETS),
                "streams": ["original", "fresh"],
                "primary_interaction": (
                    "(blocked - interleaved at rung 2) - mean(blocked - interleaved "
                    "at rungs 3 and 4)"
                ),
                "schedule_only_data_intervention": True,
            },
            indent=2,
        ) + "\n"
    )
    print(summary.to_string(index=False))
    print("\nScaling interactions\n", interactions.to_string(index=False))
    print("\nReplay versus selected conventional schedule\n", replay_summary.to_string(index=False))
    print("\nSelectors\n", selector_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
