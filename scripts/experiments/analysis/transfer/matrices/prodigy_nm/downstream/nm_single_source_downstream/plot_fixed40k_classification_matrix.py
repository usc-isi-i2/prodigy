#!/usr/bin/env python3
"""Plot matched-40k downstream CLS for specialists versus the full mixture."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SINGLES_PATH = HERE / "data/classification.csv"
LADDER_PATH = (
    HERE.parents[3]
    / "ablations/prodigy_nm/downstream/nm_ladder_downstream_nhop2/data/classification_roc_auc.csv"
)
OUTPUT_DATA = HERE / "data/classification_fixed40k_with_mixture.csv"
OUTPUT_PNG = HERE / "figures/classification_fixed40k_with_mixture.png"
OUTPUT_PDF = HERE / "figures/classification_fixed40k_with_mixture.pdf"

TARGETS = ["covid_political", "election2020", "ukr_rus_suspended", "twibot20"]
SOURCE_ORDER = [
    "ukr_rus_twitter",
    "covid19_twitter",
    "midterm",
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
    "cp_hk_twitter",
]
LABELS = {
    "ukr_rus_twitter": "UKR/RUS",
    "covid19_twitter": "COVID-19",
    "midterm": "US midterm",
    "covid_political": "COVID political",
    "election2020": "Election 2020",
    "ukr_rus_suspended": "UKR/RUS suspended",
    "twibot20": "TwiBot-20",
    "cp_hk_twitter": "CP/HK",
    "full_mixture": "Full eight-graph mixture",
    "specialist_mean": "Specialist mean",
    "specialist_median": "Specialist median",
    "specialist_max": "Specialist max",
}


def build_matrix() -> pd.DataFrame:
    singles = pd.read_csv(SINGLES_PATH).set_index("source").loc[SOURCE_ORDER, TARGETS]
    ladder = pd.read_csv(LADDER_PATH)
    mixture_rows = ladder[
        (ladder["variant"] == "matched40k")
        & (ladder["order"] == "A")
        & (ladder["rung"] == 8)
    ]
    if len(mixture_rows) != 1:
        raise ValueError(f"expected one matched-40k full-mixture row, found {len(mixture_rows)}")
    mixture = mixture_rows.set_index(pd.Index(["full_mixture"]))[TARGETS]
    summaries = pd.DataFrame(
        [singles.mean(), singles.median(), singles.max()],
        index=["specialist_mean", "specialist_median", "specialist_max"],
    )
    matrix = pd.concat([singles, mixture, summaries])
    matrix["min"] = matrix[TARGETS].min(axis=1)
    matrix["mean"] = matrix[TARGETS].mean(axis=1)
    matrix["max"] = matrix[TARGETS].max(axis=1)
    return matrix


def plot(matrix: pd.DataFrame) -> None:
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(13.8, 10.5))
    image = ax.imshow(values, cmap="viridis", vmin=0.45, vmax=1.0, aspect="auto")

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            color = "white" if value < 0.70 else "#161616"
            ax.text(col, row, f"{value:.3f}", ha="center", va="center", color=color, fontsize=9)

    ax.set_xticks(range(len(matrix.columns)), [LABELS.get(column, column.title()) for column in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), [LABELS[index] for index in matrix.index])
    ax.tick_params(axis="x", rotation=38, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlabel("Downstream classification target")
    ax.set_ylabel("Matched-40k pretraining source, mixture, or specialist summary")
    ax.axhline(len(SOURCE_ORDER) - 0.5, color="white", linewidth=3.0)
    ax.axhline(len(SOURCE_ORDER) + 0.5, color="white", linewidth=3.0)
    ax.axvline(len(TARGETS) - 0.5, color="white", linewidth=3.0)
    ax.set_title("PRODIGY matched-40k classification transfer", loc="left", fontsize=15, fontweight="bold", pad=42)
    ax.text(
        0,
        1.025,
        "Single training seed · fixed paired 10-shot episodes · every encoder receives 40k total pretraining steps",
        transform=ax.transAxes,
        color="#77746e",
        fontsize=9,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.018)
    colorbar.set_label("Classification ROC-AUC")
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    matrix = build_matrix()
    matrix.rename_axis("pretraining_source").to_csv(OUTPUT_DATA)
    plot(matrix)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
