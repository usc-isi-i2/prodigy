#!/usr/bin/env python3
"""Plot mean NM performance for interleaved, sequential, and unconfined ladders."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/prodigy-mpl-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
INPUT = HERE / "data" / "nm_ladder_schedule_comparison_long.csv"
OUTPUT = HERE / "figures" / "interleaved_vs_sequential_mean.png"


def main() -> None:
    rows = list(csv.DictReader(INPUT.open(newline="", encoding="utf-8")))
    rungs = np.arange(1, 9)
    specs = [
        ("auc_interleaved", "Balanced interleaved", "#2A78D6"),
        ("auc_sequential", "Sequential", "#D85A30"),
        ("auc_unconfined", "Merged, unconfined", "#2E8B57"),
    ]
    series = {}
    for key, _, _ in specs:
        if not all(row.get(key) not in (None, "") for row in rows):
            raise RuntimeError(f"comparison table is missing complete {key} values")
        series[key] = np.array([
            np.mean([float(row[key]) for row in rows if int(row["rung"]) == rung])
            for rung in rungs
        ])

    plt.rcParams.update({"font.size": 21, "axes.titlesize": 28, "axes.labelsize": 24,
                         "xtick.labelsize": 19, "ytick.labelsize": 19, "legend.fontsize": 19})
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    for key, label, color in specs:
        ax.plot(rungs, series[key], color=color, marker="o", linewidth=4, markersize=10, label=label)
    ax.set_xticks(rungs)
    values = np.concatenate(list(series.values()))
    padding = max(0.02, (values.max() - values.min()) * 0.12)
    ax.set_ylim(values.min() - padding, min(1.0, values.max() + padding))
    ax.set_xlabel("Ladder rung (number of source graphs)", labelpad=12)
    ax.set_ylabel("Mean NM ROC-AUC", labelpad=12)
    ax.set_title("Source scheduling across the graph ladder", pad=18)
    ax.grid(axis="y", color="#D9D9D9", linewidth=1.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout(pad=1.2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
