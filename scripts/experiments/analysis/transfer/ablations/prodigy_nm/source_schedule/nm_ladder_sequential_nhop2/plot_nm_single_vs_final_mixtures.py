#!/usr/bin/env python3
"""Compare the single-source baseline with the three final eight-source mixtures."""

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
OUTPUT = HERE / "figures" / "nm_auc_single_vs_final_mixtures.png"

# Mean of the eight rung-8 cells from the completed unconfined evaluation. The
# current branch's paired table predates its auc_unconfined column.
UNCONFINED_FINAL_MEAN = 0.8633490509758142


def rung_mean(rows: list[dict[str, str]], rung: int, field: str) -> float:
    return float(np.mean([
        float(row[field]) for row in rows if int(row["rung"]) == rung
    ]))


def main() -> None:
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    values = [
        rung_mean(rows, 1, "auc_interleaved"),
        rung_mean(rows, 8, "auc_interleaved"),
        rung_mean(rows, 8, "auc_sequential"),
        UNCONFINED_FINAL_MEAN,
    ]
    labels = [
        "Single source",
        "Balanced\ninterleaved",
        "Sequential",
        "Merged,\nunconfined",
    ]
    colors = ["#727272", "#2A78D6", "#D85A30", "#2E8B57"]

    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "xtick.labelsize": 18,
        "ytick.labelsize": 19,
    })
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    x = np.array([0.0, 1.55, 2.55, 3.55])
    bars = ax.bar(x, values, width=0.72, color=colors)

    ax.axvline(0.78, color="#C7C7C7", linewidth=1.5)
    ax.text(0.0, 0.947, "BASELINE", ha="center", va="top", fontsize=15,
            color="#666666", fontweight="bold")
    ax.text(2.55, 0.947, "FINAL 8-SOURCE MIXTURES", ha="center", va="top",
            fontsize=15, color="#666666", fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.82, 0.95)
    ax.set_ylabel("Mean NM ROC-AUC", labelpad=12)
    ax.set_title("Single-source model vs. final mixtures", pad=18)
    ax.grid(axis="y", color="#D9D9D9", linewidth=1.4)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in values],
                 padding=8, fontsize=23)

    fig.tight_layout(pad=1.2)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
