#!/usr/bin/env python3
"""Plot raw and within-model normalized native-objective losses."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/native_ssl_loss_summary.csv"
FIGURES = ROOT / "figures"
MODELS = ("PRODIGY", "VISION", "SAMGPT", "GraphSAGE")
COLORS = {
    "PRODIGY": "#4477AA", "VISION": "#228833",
    "SAMGPT": "#CC6677", "GraphSAGE": "#AA3377",
}


def load() -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    with DATA.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row["step"] = int(row["step"])
            row["loss_mean"] = float(row["loss_mean"])
            row["loss_std"] = float(row["loss_std"]) if row["loss_std"] else np.nan
            grouped[row["model"]].append(row)
    return {model: sorted(rows, key=lambda row: row["step"]) for model, rows in grouped.items()}


def draw(normalized: bool) -> None:
    grouped = load()
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), sharey=normalized)
    for index, (axis, model) in enumerate(zip(axes, MODELS)):
        rows = grouped[model]
        steps = np.array([row["step"] for row in rows])
        losses = np.array([row["loss_mean"] for row in rows])
        baseline = losses[0] if normalized else 1.0
        values = losses / baseline
        if model == "GraphSAGE":
            window = 7
            radius = window // 2
            padded = np.pad(values, (radius, radius), mode="edge")
            smoothed = np.convolve(padded, np.ones(window) / window, mode="valid")
            selected = np.linspace(0, len(steps) - 1, 6, dtype=int)
            axis.plot(
                steps[selected], smoothed[selected], color=COLORS[model], marker="o",
                markersize=4, linewidth=2.4,
            )
        else:
            axis.plot(steps, values, color=COLORS[model], marker="o", linewidth=2.2)
        axis.set_xlim(left=0)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axis.set_xlabel("Pretraining updates")
        axis.grid(axis="y", alpha=0.25)
        subtitle = rows[0]["objective"]
        if model == "GraphSAGE":
            subtitle += " (7-record centered moving average)"
        axis.set_title(f"{model}\n{subtitle}")
        if normalized:
            axis.axhline(1.0, color="#777777", linewidth=1, linestyle="--")
        if index == 0:
            axis.set_ylabel("Loss / first recorded loss" if normalized else "Native-objective loss")
    qualifier = (
        "within-model normalized; linear update scale"
        if normalized else "raw; model-specific loss and linear update scales"
    )
    fig.suptitle(f"Native-pretraining loss trajectories ({qualifier})", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    suffix = "normalized" if normalized else "raw"
    stem = FIGURES / f"native_ssl_loss_by_model_{suffix}"
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def draw_graphsage_all9() -> None:
    path = ROOT / "data/graphsage_social_all9_training_curve.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    steps = np.array([int(row["step"]) for row in rows])
    train = np.array([float(row["train_loss"]) for row in rows])
    validation = np.array([float(row["validation_loss"]) for row in rows])
    best_index = int(np.argmin(validation))

    fig, axis = plt.subplots(figsize=(9.2, 5.7))
    axis.plot(
        steps, train, color="#9c755f", marker="o", linewidth=2.0,
        label="Recorded training minibatch",
    )
    axis.plot(
        steps, validation, color="#4e79a7", marker="o", linewidth=2.8,
        label="Nine-source validation mean",
    )
    axis.scatter(
        [steps[best_index]], [validation[best_index]], s=115, marker="*",
        color="#111111", zorder=4, label=f"Best validation: step {steps[best_index]:,}",
    )
    axis.axvline(2500, color="#777777", linestyle="--", linewidth=1.5)
    axis.text(
        2480, axis.get_ylim()[1], "Fixed endpoint",
        ha="right", va="top", fontsize=10.5, color="#666666",
    )
    axis.set_xlabel("Pretraining updates", fontsize=13)
    axis.set_ylabel("Native link-prediction loss", fontsize=13)
    axis.set_title(
        "GraphSAGE all-nine social mixture: training and validation loss",
        fontsize=16, fontweight="bold", pad=10,
    )
    axis.tick_params(axis="both", labelsize=11)
    axis.grid(axis="y", color="#dddddd", linewidth=0.9)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=11, loc="upper right")
    fig.text(
        0.5, 0.012,
        "Seed 0. Training is one recorded minibatch; validation averages fixed held-out edge batches across all nine sources.",
        ha="center", fontsize=9.5, color="#666666",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    stem = FIGURES / "graphsage_social_all9_training_curve"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    draw(normalized=False)
    draw(normalized=True)
    draw_graphsage_all9()


if __name__ == "__main__":
    main()
