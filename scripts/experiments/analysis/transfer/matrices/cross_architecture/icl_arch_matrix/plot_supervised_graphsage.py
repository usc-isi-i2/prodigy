#!/usr/bin/env python3
"""Plot target-supervised GraphSAGE beside the existing architecture audit."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ARCHITECTURE_RESULTS = HERE / "data" / "raw_aggregate" / "summary" / "classification_long.csv"
SUPERVISED_RESULTS = HERE / "data" / "supervised_target" / "target_summary.csv"
OUTPUT_STEM = HERE / "figures" / "supervised_graphsage_comparison"

TARGETS = ("covid_political", "election2020", "ukr_rus_suspended", "twibot20")
TARGET_LABELS = ("Covid\nPolitical", "Election\n2020", "Ukraine\nSuspended", "TwiBot-20")
SYSTEMS = ("prodigy", "vision", "gilt", "supervised_mlp", "supervised_graphsage")
LABELS = {
    "prodigy": "PRODIGY",
    "vision": "VISION",
    "gilt": "GILT",
    "supervised_mlp": "Supervised MLP",
    "supervised_graphsage": "Supervised GraphSAGE",
}
COLORS = {
    "prodigy": "#2878B5",
    "vision": "#E07A3F",
    "gilt": "#4E9F6D",
    "supervised_mlp": "#8C8C8C",
    "supervised_graphsage": "#6B4C9A",
}


def load_results() -> pd.DataFrame:
    architecture = pd.read_csv(ARCHITECTURE_RESULTS)
    architecture = (
        architecture.groupby(["architecture", "dataset"], as_index=False)["roc_auc"]
        .mean()
        .rename(columns={"architecture": "system"})
    )
    supervised = pd.read_csv(SUPERVISED_RESULTS).rename(columns={"baseline": "system"})
    supervised = supervised[["system", "dataset", "roc_auc"]]
    results = pd.concat([architecture, supervised], ignore_index=True)

    expected = {(system, target) for system in SYSTEMS for target in TARGETS}
    observed = set(results[["system", "dataset"]].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError(f"result grid mismatch: missing={sorted(expected - observed)}")
    if not results["roc_auc"].between(0, 1).all():
        raise ValueError("ROC-AUC outside [0, 1]")
    return results


def plot(results: pd.DataFrame) -> None:
    fig, (target_ax, mean_ax) = plt.subplots(
        1,
        2,
        figsize=(12.4, 4.8),
        gridspec_kw={"width_ratios": [4.2, 1.15]},
        constrained_layout=True,
    )

    x = np.arange(len(TARGETS))
    width = 0.15
    offsets = (np.arange(len(SYSTEMS)) - (len(SYSTEMS) - 1) / 2) * width
    for offset, system in zip(offsets, SYSTEMS):
        part = results[results.system == system].set_index("dataset")
        values = [float(part.loc[target, "roc_auc"]) for target in TARGETS]
        bars = target_ax.bar(
            x + offset,
            values,
            width,
            label=LABELS[system],
            color=COLORS[system],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        if system == "supervised_graphsage":
            for bar, value in zip(bars, values):
                target_ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.014,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    fontweight="bold",
                    rotation=90,
                    color=COLORS[system],
                )

    target_ax.axhline(0.5, color="#666666", linewidth=0.9, linestyle="--", zorder=2)
    target_ax.set_xticks(x, TARGET_LABELS)
    target_ax.set_ylim(0.45, 1.04)
    target_ax.set_ylabel("ROC-AUC")
    target_ax.set_title("a  Performance by target", loc="left", fontweight="bold")

    means = results.groupby("system")["roc_auc"].mean()
    mean_values = [float(means[system]) for system in SYSTEMS]
    y = np.arange(len(SYSTEMS))
    mean_ax.barh(
        y,
        mean_values,
        color=[COLORS[system] for system in SYSTEMS],
        edgecolor="white",
        linewidth=0.45,
        zorder=3,
    )
    for row, value in zip(y, mean_values):
        mean_ax.text(value + 0.008, row, f"{value:.3f}", va="center", fontsize=8.2)
    mean_ax.axvline(0.5, color="#666666", linewidth=0.9, linestyle="--", zorder=2)
    mean_ax.set_yticks(y, [LABELS[system] for system in SYSTEMS], fontsize=8.5)
    mean_ax.invert_yaxis()
    mean_ax.set_xlim(0.45, 0.89)
    mean_ax.set_xlabel("Mean ROC-AUC")
    mean_ax.set_title("b  Four-target mean", loc="left", fontweight="bold")

    for axis in (target_ax, mean_ax):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y" if axis is target_ax else "x", color="#dddddd", linewidth=0.6, zorder=0)

    handles, legend_labels = target_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        ncols=5,
        fontsize=8.5,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.suptitle(
        "Target-supervised GraphSAGE reference at 100 updates",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    fig.text(
        0.5,
        -0.02,
        "Supervised references use target training labels; PRODIGY, VISION, and GILT are 10-shot in-context systems.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot(load_results())
    print(OUTPUT_STEM.with_suffix(".png"))
    print(OUTPUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
