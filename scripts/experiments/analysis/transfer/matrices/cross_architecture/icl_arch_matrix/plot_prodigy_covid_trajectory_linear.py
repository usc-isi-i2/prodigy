#!/usr/bin/env python3
"""Plot the PRODIGY COVID-political trajectory on a true linear x-axis."""

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
INPUT = HERE / "data" / "native_source_900_seed0" / "classification_all.tsv"
OUTPUT = HERE / "figures" / "prodigy_native_nm_covid_political_linear.png"
STEPS = (0, 60, 100, 300, 900)


def main() -> None:
    with INPUT.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    values = [
        float(np.mean([
            float(row["roc_auc"])
            for row in rows
            if row["architecture"] == "prodigy"
            and row["dataset"] == "covid_political"
            and int(row["checkpoint_step"]) == step
        ]))
        for step in STEPS
    ]

    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 24,
        "axes.labelsize": 21,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.plot(
        STEPS, values, color="#4477AA", marker="o",
        markersize=8, linewidth=3,
    )
    ax.set_xticks(STEPS, [str(step) for step in STEPS], rotation=35, ha="right")
    ax.set_xlim(-20, 930)
    ax.set_ylim(0.35, 0.90)
    ax.set_xlabel("Training checkpoint")
    ax.set_ylabel("NM ROC-AUC")
    ax.set_title("PRODIGY native NM pretraining: COVID political")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
