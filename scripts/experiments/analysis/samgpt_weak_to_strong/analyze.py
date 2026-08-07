#!/usr/bin/env python3
"""Summarize the SAMGPT weak-to-strong held-out TwiBot-20 ladder."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PREFIXES = [
    ["ukr_rus_suspended"],
    ["ukr_rus_suspended", "covid_political"],
    ["ukr_rus_suspended", "covid_political", "election2020"],
    [
        "ukr_rus_suspended",
        "covid_political",
        "election2020",
        "midterm",
    ],
    [
        "ukr_rus_suspended",
        "covid_political",
        "election2020",
        "midterm",
        "cp_hk_twitter",
    ],
]
SOURCE_LABELS = [
    "ukr suspended",
    "+ covid political",
    "+ election2020",
    "+ midterm",
    "+ cp_hk",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=39)
    parser.add_argument("--specialist-summary", type=Path, default=None)
    return parser.parse_args()


def read_episode_column(path: Path, column: str) -> np.ndarray:
    with path.open(newline="") as handle:
        return np.asarray(
            [float(row[column]) for row in csv.DictReader(handle)], dtype=np.float64
        )


def bootstrap_mean_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    output_data = args.output_root / "data"
    output_figures = args.output_root / "figures"
    output_data.mkdir(parents=True, exist_ok=True)
    output_figures.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    auc_episodes: list[np.ndarray] = []
    accuracy_episodes: list[np.ndarray] = []
    for rung, sources in enumerate(PREFIXES, start=1):
        rung_dir = args.input_root / f"r{rung}"
        result = json.loads((rung_dir / "metrics.json").read_text())
        metrics = result["metrics"]
        training = result["pretraining"]
        summaries.append(
            {
                "rung": rung,
                "n_sources": len(sources),
                "source_added": sources[-1],
                "sources": ";".join(sources),
                "roc_auc": metrics["roc_auc_mean"],
                "accuracy": metrics["accuracy_mean"],
                "epochs": training["epochs_run"],
                "training_seconds": training["seconds"],
                "best_loss": training["best_loss"],
                "run_name": Path(training["checkpoint"]).parent.name,
                "reused_existing_run": rung in (1, 5),
            }
        )
        auc_episodes.append(read_episode_column(rung_dir / "episode_metrics.csv", "roc_auc"))
        accuracy_episodes.append(
            read_episode_column(rung_dir / "episode_metrics.csv", "accuracy")
        )

    write_csv(output_data / "rung_summary.csv", summaries)

    delta_rows: list[dict[str, object]] = []
    for metric_name, episode_values in (
        ("roc_auc", auc_episodes),
        ("accuracy", accuracy_episodes),
    ):
        comparisons = [
            (f"r{index + 2}-r{index + 1}", index, index + 1)
            for index in range(4)
        ] + [("r5-r1", 0, 4)]
        for comparison, left, right in comparisons:
            delta = episode_values[right] - episode_values[left]
            lower, upper = bootstrap_mean_interval(delta, rng, args.bootstrap_samples)
            delta_rows.append(
                {
                    "metric": metric_name,
                    "comparison": comparison,
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(np.median(delta)),
                    "positive_episode_fraction": float(np.mean(delta > 0)),
                    "bootstrap_95_lower": lower,
                    "bootstrap_95_upper": upper,
                    "episodes": len(delta),
                }
            )
    write_csv(output_data / "paired_deltas.csv", delta_rows)

    specialist_path = args.specialist_summary or output_data / "specialist_summary.csv"
    with specialist_path.open(newline="") as handle:
        specialists = {
            row["source"]: float(row["roc_auc"]) for row in csv.DictReader(handle)
        }

    max_rule_rows: list[dict[str, object]] = []
    for summary, sources in zip(summaries, PREFIXES):
        best_source = max(sources, key=specialists.__getitem__)
        prediction = specialists[best_source]
        observed = float(summary["roc_auc"])
        max_rule_rows.append(
            {
                "rung": summary["rung"],
                "source_added": summary["source_added"],
                "observed_roc_auc": observed,
                "max_rule_prediction": prediction,
                "residual_observed_minus_prediction": observed - prediction,
                "cumulative_best_source": best_source,
            }
        )
    write_csv(output_data / "max_rule_comparison.csv", max_rule_rows)

    residuals = np.asarray(
        [float(row["residual_observed_minus_prediction"]) for row in max_rule_rows]
    )
    write_csv(
        output_data / "max_rule_metrics.csv",
        [
            {
                "cells": len(residuals),
                "mae": float(np.mean(np.abs(residuals))),
                "rmse": float(np.sqrt(np.mean(residuals**2))),
                "max_absolute_error": float(np.max(np.abs(residuals))),
                "mean_signed_residual": float(np.mean(residuals)),
            }
        ],
    )

    observed = np.asarray([float(row["roc_auc"]) for row in summaries])
    predicted = np.asarray(
        [float(row["max_rule_prediction"]) for row in max_rule_rows]
    )
    x = np.arange(1, 6)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    axes[0].plot(
        x,
        observed,
        marker="o",
        linewidth=2.2,
        color="#2367a8",
        label="Observed mixture",
    )
    axes[0].plot(
        x,
        predicted,
        marker="s",
        linestyle="--",
        linewidth=1.8,
        color="#6f4b8b",
        label="Cumulative best specialist",
    )
    axes[0].set_xticks(x, SOURCE_LABELS, rotation=24, ha="right")
    axes[0].set_ylabel("TwiBot-20 ROC-AUC")
    axes[0].set_title("Weak-to-strong donor order")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8.5)

    colors = ["#c2572a" if value > 0 else "#2367a8" for value in residuals]
    axes[1].bar(x, residuals, color=colors, width=0.62)
    axes[1].axhline(0, color="#444444", linewidth=1)
    axes[1].set_xticks(x, SOURCE_LABELS, rotation=24, ha="right")
    axes[1].set_ylabel("Observed minus max-rule prediction")
    axes[1].set_title("Composition-rule residual")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(
        "SAMGPT weak-to-strong source-mixture ladder · seed 39",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "C1 and C5 reuse source-slot-aligned runs; all rungs use the same 500 evaluation episodes.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(output_figures / "mixture_weak_to_strong.png", dpi=220)
    fig.savefig(output_figures / "mixture_weak_to_strong.pdf")


if __name__ == "__main__":
    main()
