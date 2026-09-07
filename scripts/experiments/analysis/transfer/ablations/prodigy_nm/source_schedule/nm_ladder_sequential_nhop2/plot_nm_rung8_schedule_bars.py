#!/usr/bin/env python3
"""Plot rung-8 NM ROC-AUC by graph for interleaved and sequential schedules."""

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
INPUT = HERE / "data" / "nm_ladder_schedule_comparison_long.csv"
OUTPUT = HERE / "figures" / "nm_auc_interleaved_vs_sequential_rung8.png"
GRAPHS = [
    "ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
    "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter",
]
LABELS = [
    "Ukr-Rus", "COVID-19", "Midterm", "COVID-pol.",
    "Election '20", "Ukr-Rus\nsusp.", "TwiBot-20", "CP-HK",
]


def main() -> None:
    rows = [
        row for row in csv.DictReader(INPUT.open(newline="", encoding="utf-8"))
        if int(row["rung"]) == 8
    ]
    by_graph = {row["test_graph"]: row for row in rows}
    interleaved = np.array([float(by_graph[graph]["auc_interleaved"]) for graph in GRAPHS])
    sequential = np.array([float(by_graph[graph]["auc_sequential"]) for graph in GRAPHS])

    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "xtick.labelsize": 16,
        "ytick.labelsize": 19,
        "legend.fontsize": 21,
    })
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    values = [interleaved.mean(), sequential.mean()]
    x = np.arange(2)
    bars = ax.bar(x, values, width=0.58, color=["#2A78D6", "#D85A30"])
    ax.set_xticks(x, ["Balanced interleaved", "Sequential"])
    ax.set_xlim(-0.8, 1.8)
    ax.set_ylim(0.75, 0.95)
    ax.set_ylabel("Mean NM ROC-AUC", labelpad=12)
    ax.set_title("Eight-graph mixture: mean NM performance", pad=18)
    ax.grid(axis="y", color="#D9D9D9", linewidth=1.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=8, fontsize=24)
    fig.tight_layout(pad=1.2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
