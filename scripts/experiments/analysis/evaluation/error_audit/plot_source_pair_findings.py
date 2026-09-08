#!/usr/bin/env python3
"""Render the committed figures for the source-paired error-audit reports."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    }
)


def covid_classification() -> None:
    cohorts = [
        "Both TN",
        "HK FP / UKR TN",
        "UKR FP / HK TN",
        "Both TP",
        "HK FN / UKR TP",
        "Both FN",
    ]
    isolated_original = np.array([45.8, 10.2, 71.5, 8.4, 52.7, 73.6])
    isolated_fresh = np.array([45.2, 9.5, 78.3, 8.1, 51.8, 77.5])
    incoming_counts = np.array(
        [
            [722, 339, 158, 89, 48],
            [78, 267, 177, 81, 54],
            [109, 19, 13, 2, 1],
            [48, 136, 164, 118, 92],
            [92, 18, 16, 13, 11],
        ]
    )
    outcomes = np.array([[1914, 200, 807, 151], [1989, 172, 762, 149]])

    fig = plt.figure(figsize=(11.5, 8.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 0.8], hspace=0.62, wspace=0.32)
    ax_iso = fig.add_subplot(grid[0, 0])
    ax_degree = fig.add_subplot(grid[0, 1])
    ax_outcome = fig.add_subplot(grid[1, :])

    y = np.arange(len(cohorts))
    ax_iso.hlines(y, isolated_original, isolated_fresh, color="0.75", linewidth=2)
    ax_iso.scatter(isolated_original, y, marker="o", label="Original episodes", zorder=3)
    ax_iso.scatter(isolated_fresh, y, marker="s", label="Fresh episodes", zorder=3)
    ax_iso.set_yticks(y, cohorts)
    ax_iso.invert_yaxis()
    ax_iso.set_xlim(0, 85)
    ax_iso.set_xlabel("Queries with zero full-graph degree (%)")
    ax_iso.set_title("Isolation sharply separates error cohorts")
    ax_iso.grid(axis="x", alpha=0.25)
    ax_iso.legend(frameon=False, loc="lower right")

    degree_labels = ["0", "1", "2–3", "4–7", "8+"]
    degree_cohorts = cohorts[:5]
    shares = incoming_counts / incoming_counts.sum(axis=1, keepdims=True)
    left = np.zeros(len(degree_cohorts))
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(degree_labels)))
    for label, values, color in zip(degree_labels, shares.T, colors):
        ax_degree.barh(degree_cohorts, values * 100, left=left * 100, label=label, color=color)
        left += values
    ax_degree.invert_yaxis()
    ax_degree.set_xlim(0, 100)
    ax_degree.set_xlabel("Share of cohort in original stream (%)")
    ax_degree.set_title("Incoming-degree distribution")
    ax_degree.legend(
        title="Incoming degree", ncol=5, frameon=False, fontsize=8,
        loc="upper center", bbox_to_anchor=(0.5, -0.20)
    )

    outcome_labels = ["Both correct", "Both wrong", "UKR only correct", "HK only correct"]
    outcome_colors = plt.cm.Set2(np.linspace(0.05, 0.85, 4))
    left = np.zeros(2)
    for label, values, color in zip(outcome_labels, outcomes.T, outcome_colors):
        ax_outcome.barh(["Original", "Fresh"], values, left=left, label=label, color=color)
        for row, value in enumerate(values):
            if value >= 170:
                ax_outcome.text(left[row] + value / 2, row, f"{value:,}", ha="center", va="center", fontsize=9)
        left += values
    ax_outcome.invert_yaxis()
    ax_outcome.set_xlim(0, 3072)
    ax_outcome.set_xlabel("Paired query occurrences")
    ax_outcome.set_title("Source-specific correctness replicates across episode draws")
    ax_outcome.legend(ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.48))

    fig.suptitle("COVID Political classification: where source specialists disagree", fontsize=14, y=0.98)
    fig.savefig(FIGURES / "covid_political_fpfn_distributions.png")
    plt.close(fig)


def neighbor_matching() -> None:
    targets = ["Ukraine test", "Ukraine validation", "Hong Kong test", "Hong Kong validation"]
    ukr = np.array([42.678, 42.509, 13.203, 13.375])
    hk = np.array([11.357, 11.653, 20.255, 19.945])
    oracle = np.array([46.432, 46.213, 26.988, 26.785])
    test_cohorts = {
        "Ukraine test": [7.603, 53.568, 35.075, 3.754],
        "Hong Kong test": [6.470, 73.012, 6.733, 13.785],
    }

    fig, (ax_acc, ax_overlap) = plt.subplots(1, 2, figsize=(11.5, 4.9), layout="constrained")
    y = np.arange(len(targets))
    height = 0.23
    ax_acc.barh(y - height, ukr, height, label="Ukraine model")
    ax_acc.barh(y, hk, height, label="Hong Kong model")
    ax_acc.barh(y + height, oracle, height, label="Either-model oracle", color="0.55")
    ax_acc.set_yticks(y, targets)
    ax_acc.invert_yaxis()
    ax_acc.set_xlim(0, 50)
    ax_acc.set_xlabel("Query accuracy (%)")
    ax_acc.set_title("Native-source specialization replicates")
    ax_acc.grid(axis="x", alpha=0.25)
    ax_acc.legend(frameon=False, loc="lower right")

    labels = list(test_cohorts)
    vals = np.array(list(test_cohorts.values()))
    cohort_labels = ["Both correct", "Both wrong", "UKR only correct", "HK only correct"]
    colors = plt.cm.Set2(np.linspace(0.05, 0.85, 4))
    left = np.zeros(len(labels))
    for label, values, color in zip(cohort_labels, vals.T, colors):
        ax_overlap.barh(labels, values, left=left, label=label, color=color)
        for row, value in enumerate(values):
            if value >= 5:
                ax_overlap.text(left[row] + value / 2, row, f"{value:.1f}%", ha="center", va="center", fontsize=9)
        left += values
    ax_overlap.invert_yaxis()
    ax_overlap.set_xlim(0, 100)
    ax_overlap.set_xlabel("Paired query occurrences (%)")
    ax_overlap.set_title("A shared hard core plus source-specific rescues")
    ax_overlap.legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.36))

    fig.suptitle("30-way neighbor matching: paired model outcomes", fontsize=14)
    fig.savefig(FIGURES / "nm_source_pair_outcomes.png")
    plt.close(fig)


if __name__ == "__main__":
    covid_classification()
    neighbor_matching()
