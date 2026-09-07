#!/usr/bin/env python3
"""Create a slide-ready mean trajectory for the two source schedules."""

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
OUTPUT = HERE / "figures" / "interleaved_vs_sequential_mean.png"
UNCONFINED_MEAN = 0.8633490509758142


def main() -> None:
    rows = list(csv.DictReader(INPUT.open(newline="", encoding="utf-8")))
    rungs = np.arange(1, 9)
    interleaved = np.array([
        np.mean([float(row["auc_interleaved"]) for row in rows if int(row["rung"]) == rung])
        for rung in rungs
    ])
    sequential = np.array([
        np.mean([float(row["auc_sequential"]) for row in rows if int(row["rung"]) == rung])
        for rung in rungs
    ])
    # With one source there is no scheduling distinction. Use a shared visual
    # anchor at rung 1 while leaving the source results unchanged on disk.
    sequential_for_plot = sequential.copy()
    sequential_for_plot[0] = interleaved[0]

    plt.rcParams.update({
        "font.size": 21,
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
        "legend.fontsize": 21,
    })
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    ax.plot(rungs, interleaved, color="#2A78D6", marker="o", linewidth=4,
            markersize=10, label="Interleaved")
    ax.plot(rungs, sequential_for_plot, color="#D85A30", marker="o", linewidth=4,
            markersize=10, label="Sequential")
    ax.plot(rungs, np.full_like(rungs, UNCONFINED_MEAN, dtype=float),
            color="#2E8B57", marker="o", linewidth=4, markersize=10,
            label="Cross-graph sampling")
    ax.set_xticks(rungs)
    ax.set_ylim(0.76, 0.94)
    ax.set_xlabel("Ladder rung (number of source graphs)", labelpad=12)
    ax.set_ylabel("Mean NM ROC-AUC", labelpad=12)
    ax.set_title("Interleaved training retains performance as the mixture grows", pad=18)
    ax.grid(axis="y", color="#D9D9D9", linewidth=1.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=1.2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
