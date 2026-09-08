#!/usr/bin/env python3
"""Compare single-graph and final-mixture label-classification ROC-AUC."""

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
SPECIALIST_INPUT = (
    TRANSFER_ROOT / "matrices" / "prodigy_nm" / "downstream"
    / "nm_single_source_downstream" / "data" / "classification.csv"
)
MIXTURE_INPUT = (
    HERE.parents[1] / "downstream" / "nm_ladder_downstream_nhop2"
    / "data" / "downstream_long.csv"
)
OUTPUT = HERE / "figures" / "cls_auc_single_graph_model_means.png"
GRAPHS = [
    "ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
    "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter",
]
LABEL_BY_GRAPH = {
    "ukr_rus_twitter": "Ukr-Rus",
    "covid19_twitter": "COVID-19",
    "midterm": "Midterm",
    "covid_political": "COVID-pol.",
    "election2020": "Election '20",
    "ukr_rus_suspended": "Ukr-Rus\nsusp.",
    "twibot20": "TwiBot-20",
    "cp_hk_twitter": "CP-HK",
}


def main() -> None:
    with SPECIALIST_INPUT.open(newline="", encoding="utf-8") as handle:
        specialist_rows = {row["source"]: row for row in csv.DictReader(handle)}
    specialist_values = np.array([
        float(specialist_rows[graph]["mean"]) for graph in GRAPHS
    ])
    order = np.argsort(specialist_values)[::-1]
    specialist_values = specialist_values[order]
    specialist_labels = [LABEL_BY_GRAPH[GRAPHS[index]] for index in order]

    with MIXTURE_INPUT.open(newline="", encoding="utf-8") as handle:
        mixture_rows = [
            row for row in csv.DictReader(handle)
            if row["task"] == "classification"
            and row["metric"] == "roc_auc"
            and row["variant"] in {"matched40k", "sequential"}
            and int(row["rung"]) == 8
        ]
    mixture_values = np.array([
        np.mean([float(row["value"]) for row in mixture_rows
                 if row["variant"] == variant])
        for variant in ("matched40k", "sequential")
    ])
    mixture_labels = ["Balanced\ninterleaved", "Sequential"]

    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "xtick.labelsize": 13,
        "ytick.labelsize": 19,
    })
    fig, ax = plt.subplots(figsize=(16, 7.5))
    single_x = np.arange(len(GRAPHS))
    mixture_x = np.arange(len(GRAPHS) + 1, len(GRAPHS) + 3)
    single_bars = ax.bar(single_x, specialist_values, width=0.72, color="#777777")
    mixture_bars = ax.bar(
        mixture_x, mixture_values, width=0.72, color=["#2A78D6", "#D85A30"]
    )

    ax.axvline(len(GRAPHS) - 0.05, color="#C7C7C7", linewidth=1.5)
    ax.text(3.5, 0.985, "SINGLE-GRAPH MODELS", ha="center", va="top",
            fontsize=15, color="#666666", fontweight="bold")
    ax.text(8.5, 0.985, "FINAL 8-SOURCE MIXTURES", ha="center", va="top",
            fontsize=15, color="#666666", fontweight="bold")
    ax.set_xticks(
        np.concatenate([single_x, mixture_x]),
        specialist_labels + mixture_labels,
        rotation=24,
        ha="right",
    )
    ax.set_ylim(0.5, 1.0)
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.set_ylabel("Mean classification ROC-AUC", labelpad=12)
    ax.set_xlabel("Pre-training source configuration", labelpad=12)
    ax.set_title("Single-graph models vs. final mixtures: label classification", pad=18)
    ax.grid(axis="y", color="#D9D9D9", linewidth=1.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar_label(single_bars,
                 labels=[f"{value:.3f}" for value in specialist_values],
                 padding=7, fontsize=15)
    ax.bar_label(mixture_bars,
                 labels=[f"{value:.3f}" for value in mixture_values],
                 padding=7, fontsize=15)

    fig.tight_layout(pad=1.2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
