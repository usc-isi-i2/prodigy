#!/usr/bin/env python3
"""Compare single-graph NM models with the three final eight-source mixtures."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/prodigy-mpl-cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
TRANSFER_ROOT = HERE.parents[3]
INPUT = (
    TRANSFER_ROOT / "matrices" / "prodigy_nm" / "single_source"
    / "nm_single_source_matrix" / "data" / "nm_single_source_matrix.csv"
)
SCHEDULE_INPUT = HERE / "data" / "nm_ladder_schedule_comparison_long.csv"
OUTPUT = HERE / "figures" / "nm_auc_single_graph_model_means.png"
UNCONFINED_FINAL_MEAN = 0.8633490509758142
GRAPHS = [
    "ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
    "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter",
]
LABELS = [
    "Ukr-Rus", "COVID-19", "Midterm", "COVID-pol.",
    "Election '20", "Ukr-Rus\nsusp.", "TwiBot-20", "CP-HK",
]


def main() -> None:
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = {row["train_graph"]: row for row in csv.DictReader(handle)}

    missing = set(GRAPHS) - set(rows)
    if missing:
        raise ValueError(f"single-source matrix is missing models: {sorted(missing)}")
    values = np.array([
        np.mean([float(rows[graph][target]) for target in GRAPHS])
        for graph in GRAPHS
    ])
    single_source_mean = float(np.mean(values))

    with SCHEDULE_INPUT.open(newline="", encoding="utf-8") as handle:
        schedule_rows = list(csv.DictReader(handle))
    rung8 = [row for row in schedule_rows if int(row["rung"]) == 8]
    mixture_values = np.array([
        np.mean([float(row["auc_interleaved"]) for row in rung8]),
        np.mean([float(row["auc_sequential"]) for row in rung8]),
        UNCONFINED_FINAL_MEAN,
    ])
    mixture_labels = ["Interleaved", "Sequential", "Merged"]

    plt.rcParams.update({
        "font.size": 24,
        "axes.titlesize": 32,
        "axes.labelsize": 28,
        "xtick.labelsize": 18,
        "ytick.labelsize": 22,
    })
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    all_values = np.concatenate([[single_source_mean], mixture_values])
    all_labels = ["Mean Single-Source"] + mixture_labels
    all_colors = ["#777777", "#2A78D6", "#D85A30", "#2E8B57"]
    order = np.argsort(all_values)
    all_values = all_values[order]
    all_labels = [all_labels[index] for index in order]
    all_colors = [all_colors[index] for index in order]
    x = np.arange(len(all_values))
    bars = ax.bar(x, all_values, width=0.52, color=all_colors)
    ax.set_xticks(x, all_labels)
    ax.set_ylim(0.6, 1.0)
    ax.set_yticks(np.arange(0.6, 1.01, 0.1))
    ax.set_ylabel("Mean NM ROC-AUC", labelpad=12)
    ax.set_xlabel("Pre-training source configuration", labelpad=12)
    ax.set_title("Single-source mean vs. final mixtures", pad=18)
    ax.grid(axis="y", color="#D9D9D9", linewidth=1.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar_label(
        bars, labels=[f"{value:.3f}" for value in all_values],
        padding=7, fontsize=20,
    )

    fig.tight_layout(pad=1.2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
