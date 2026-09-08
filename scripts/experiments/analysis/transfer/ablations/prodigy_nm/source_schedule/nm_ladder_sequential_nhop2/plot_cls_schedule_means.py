#!/usr/bin/env python3
"""Plot mean classification ROC-AUC by labeled graph and source schedule."""

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
INPUT = HERE.parents[1] / "downstream" / "nm_ladder_downstream_nhop2" / "data" / "downstream_long.csv"
OUTPUT = HERE / "figures" / "cls_auc_interleaved_vs_sequential_mean.png"
GRAPHS = ["covid_political", "election2020", "ukr_rus_suspended", "twibot20"]
LABELS = ["COVID-pol.", "Election '20", "Ukr-Rus susp.", "TwiBot-20"]


def main() -> None:
    rows = [
        row for row in csv.DictReader(INPUT.open(newline="", encoding="utf-8"))
        if row["task"] == "classification" and row["metric"] == "roc_auc"
        and row["variant"] in {"matched40k", "sequential"}
        and int(row["rung"]) == 8
    ]
    means = {}
    for variant in ("matched40k", "sequential"):
        means[variant] = np.array([
            np.mean([
                float(row["value"]) for row in rows
                if row["variant"] == variant and row["dataset"] == graph
            ])
            for graph in GRAPHS
        ])

    plt.rcParams.update({
        "font.size": 21,
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
        "legend.fontsize": 21,
    })
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    x = np.arange(len(GRAPHS))
    width = 0.36
    ax.bar(x - width / 2, means["matched40k"], width, color="#2A78D6", label="Balanced interleaved")
    ax.bar(x + width / 2, means["sequential"], width, color="#D85A30", label="Sequential")
    ax.set_xticks(x, LABELS)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("Mean classification ROC-AUC", labelpad=12)
    ax.set_title("Eight-graph mixture: interleaved vs. sequential pre-training", pad=18)
    ax.grid(axis="y", color="#D9D9D9", linewidth=1.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(pad=1.2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
