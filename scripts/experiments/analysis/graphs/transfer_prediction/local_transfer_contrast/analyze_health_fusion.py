#!/usr/bin/env python3
"""Evaluate untuned TRACE probability fusion over the nine-source expert bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
import torch

from .analyze_contrast import auc_inputs, softmax, to_numpy, validate_pair
from .analyze_multi_health import (
    SUPPORT_COMPETENCE_THRESHOLD,
    health_probability_fusion,
)


TARGETS = (
    "covid_political",
    "election2020",
    "facebook_page_reference",
    "twibot20",
    "ukr_rus_suspended",
)
METHODS = (
    "discovery_auc_fixed",
    "equal_logit",
    "equal_probability",
    "trace_health_fusion",
)


def parse_seed_input(value):
    seed, path = value.split("=", 1)
    return int(seed), Path(path)


def metrics(record, probabilities):
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-12):
        raise ValueError("probabilities do not sum to one")
    y = to_numpy(record["labels"]["local_y"]).astype(int)
    auc_y, auc_score = auc_inputs(record, probabilities)
    return {
        "accuracy": float(accuracy_score(y, probabilities.argmax(axis=1))),
        "roc_auc": float(roc_auc_score(auc_y, auc_score)),
        "nll": float(log_loss(y, probabilities, labels=[0, 1])),
    }


def u1_health(record, models):
    return {
        model: (
            to_numpy(record["models"][model]["logits"]["full_model"]).argmax(1)
            == to_numpy(
                record["models"][model]["logits"]["U1_pre_meta/ridge"]
            ).argmax(1)
        )
        for model in models
    }


def support_competence(record, models):
    return {
        model: float(
            to_numpy(
                record["models"][model]["support_health"]
                ["u1_loo_prototype_accuracy"]
            ).mean()
        )
        for model in models
    }


def analyze_target(seed, target, directory):
    discovery = torch.load(
        directory / target / "original.pt", map_location="cpu", weights_only=False
    )
    validation = torch.load(
        directory / target / "fresh.pt", map_location="cpu", weights_only=False
    )
    validate_pair(discovery, validation)
    models = sorted(discovery["models"])
    if len(models) != 9 or set(validation["models"]) != set(models):
        raise ValueError(f"expected nine matched experts for seed {seed}/{target}")
    discovery_auc = {}
    for model in models:
        probabilities = softmax(
            to_numpy(discovery["models"][model]["logits"]["full_model"])
        )
        auc_y, auc_score = auc_inputs(discovery, probabilities)
        discovery_auc[model] = float(roc_auc_score(auc_y, auc_score))
    best = max(models, key=discovery_auc.get)
    competence = support_competence(validation, models)
    supported = max(competence.values()) >= SUPPORT_COMPETENCE_THRESHOLD
    rows = []
    fingerprints = {}
    for stream, record in (("original", discovery), ("fresh", validation)):
        logits = np.stack(
            [
                to_numpy(record["models"][model]["logits"]["full_model"])
                .astype(np.float64)
                for model in models
            ]
        )
        probabilities = np.stack([softmax(values) for values in logits])
        health = u1_health(record, models)
        healthy_count = np.stack([health[model] for model in models]).sum(axis=0)
        outputs = {
            "discovery_auc_fixed": probabilities[models.index(best)],
            "equal_logit": softmax(logits.mean(axis=0)),
            "equal_probability": probabilities.mean(axis=0),
            "trace_health_fusion": health_probability_fusion(
                record, models, health
            ),
        }
        for method in METHODS:
            rows.append(
                {
                    "seed": seed,
                    "target": target,
                    "stream": stream,
                    "method": method,
                    "models_evaluated": 1 if method == "discovery_auc_fixed" else 9,
                    "uses_target_query_labels": method == "discovery_auc_fixed",
                    "discovery_best_model": best,
                    "target_support_competence": max(competence.values()),
                    "target_supported_at_0_55": supported,
                    "mean_healthy_experts": float(healthy_count.mean()),
                    "no_healthy_expert_fraction": float((healthy_count == 0).mean()),
                    **metrics(record, outputs[method]),
                }
            )
        fingerprints[stream] = record["receipts"]["episode_fingerprint"]
    if fingerprints["original"] == fingerprints["fresh"]:
        raise ValueError(f"episode streams are not distinct for seed {seed}/{target}")
    return rows, fingerprints


def method_contrasts(results):
    keys = ["seed", "target"]
    comparisons = (
        ("equal_logit", "discovery_auc_fixed"),
        ("equal_probability", "discovery_auc_fixed"),
        ("trace_health_fusion", "discovery_auc_fixed"),
        ("trace_health_fusion", "equal_probability"),
    )
    rows = []
    for stream in ("original", "fresh"):
        part = results[results.stream.eq(stream)]
        indexed = {
            method: part[part.method.eq(method)].set_index(keys)
            for method in METHODS
        }
        for method, reference in comparisons:
            for key in indexed[reference].index:
                changed = indexed[method].loc[key]
                baseline = indexed[reference].loc[key]
                rows.append(
                    dict(zip(keys, key))
                    | {
                        "stream": stream,
                        "method": method,
                        "reference": reference,
                        "target_supported_at_0_55": bool(
                            changed.target_supported_at_0_55
                        ),
                        **{
                            f"delta_{metric}": float(
                                changed[metric] - baseline[metric]
                            )
                            for metric in ("accuracy", "roc_auc", "nll")
                        },
                    }
                )
    return pd.DataFrame(rows)


def crossed_bootstrap(frame, column, seed, draws=10000):
    targets = sorted(frame.target.unique())
    seeds = sorted(frame.seed.unique())
    matrix = frame.pivot(index="target", columns="seed", values=column).loc[
        targets, seeds
    ].to_numpy()
    if not np.isfinite(matrix).all():
        raise ValueError("fusion bootstrap matrix is incomplete")
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        target_indices = rng.integers(0, len(targets), len(targets))
        seed_indices = rng.integers(0, len(seeds), len(seeds))
        samples[index] = matrix[target_indices][:, seed_indices].mean()
    return float(matrix.mean()), np.quantile(samples, (0.025, 0.975))


def gain_summary(contrasts, draws=10000):
    rows = []
    for stream in ("original", "fresh"):
        stream_frame = contrasts[contrasts.stream.eq(stream)]
        for scope, scoped in (
            ("all_targets", stream_frame),
            (
                "supported_targets",
                stream_frame[stream_frame.target_supported_at_0_55],
            ),
        ):
            for comparison_index, (method, reference) in enumerate(
                sorted(set(zip(scoped.method, scoped.reference)))
            ):
                part = scoped[
                    scoped.method.eq(method) & scoped.reference.eq(reference)
                ]
                for metric_index, metric in enumerate(("accuracy", "roc_auc")):
                    column = f"delta_{metric}"
                    value, interval = crossed_bootstrap(
                        part,
                        column,
                        12100 + 100 * comparison_index + metric_index
                        + 1000 * (scope == "supported_targets")
                        + 10000 * (stream == "fresh"),
                        draws,
                    )
                    rows.append(
                        {
                            "stream": stream,
                            "scope": scope,
                            "method": method,
                            "reference": reference,
                            "metric": metric,
                            "cells": len(part),
                            "mean_delta": value,
                            "cell_wins": int((part[column] > 0).sum()),
                            "cell_ties": int((part[column] == 0).sum()),
                            "cell_losses": int((part[column] < 0).sum()),
                            "crossed_seed_target_bootstrap_low": float(interval[0]),
                            "crossed_seed_target_bootstrap_high": float(interval[1]),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-inputs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    parsed = [parse_seed_input(value) for value in args.seed_inputs]
    if sorted(seed for seed, _ in parsed) != [0, 1, 2]:
        raise ValueError("exactly checkpoint seeds 0, 1, and 2 are required")
    output_files = (
        "health_fusion_results.csv",
        "health_fusion_contrasts.csv",
        "health_fusion_gain_summary.csv",
        "health_fusion_by_target.csv",
        "health_fusion_protocol.json",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in output_files):
        raise ValueError("refusing to overwrite an existing health-fusion analysis")
    rows = []
    fingerprints = []
    for seed, directory in parsed:
        if not (directory / "DONE.json").is_file():
            raise ValueError(f"incomplete seed export: {directory}")
        for target in TARGETS:
            target_rows, target_fingerprints = analyze_target(seed, target, directory)
            rows.extend(target_rows)
            fingerprints.append(
                {"seed": seed, "target": target, **target_fingerprints}
            )
    results = pd.DataFrame(rows)
    if len(results) != 120:
        raise ValueError(f"expected 120 method cells, got {len(results)}")
    contrasts = method_contrasts(results)
    gains = gain_summary(contrasts, args.bootstrap_draws)
    by_target = contrasts.groupby(
        ["stream", "target", "method", "reference"]
    ).agg(
        seeds=("seed", "nunique"),
        mean_delta_accuracy=("delta_accuracy", "mean"),
        accuracy_wins=("delta_accuracy", lambda values: int((values > 0).sum())),
        accuracy_ties=("delta_accuracy", lambda values: int((values == 0).sum())),
        mean_delta_roc_auc=("delta_roc_auc", "mean"),
        auc_wins=("delta_roc_auc", lambda values: int((values > 0).sum())),
    ).reset_index()
    results.to_csv(args.output / output_files[0], index=False)
    contrasts.to_csv(args.output / output_files[1], index=False)
    gains.to_csv(args.output / output_files[2], index=False)
    by_target.to_csv(args.output / output_files[3], index=False)
    (args.output / output_files[4]).write_text(
        json.dumps(
            {
                "checkpoint_seeds": [0, 1, 2],
                "targets": list(TARGETS),
                "experts": 9,
                "streams": ["original", "fresh"],
                "fusion": (
                    "equal production probabilities among experts whose final "
                    "decision agrees with their U1 support-fitted ridge decision; "
                    "equal-all fallback when none agree"
                ),
                "fusion_uses_target_query_labels": False,
                "fixed_reference": "best original-stream target AUC per seed",
                "support_competence_threshold": SUPPORT_COMPETENCE_THRESHOLD,
                "fingerprints": fingerprints,
            },
            indent=2,
        ) + "\n"
    )
    print(gains.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
