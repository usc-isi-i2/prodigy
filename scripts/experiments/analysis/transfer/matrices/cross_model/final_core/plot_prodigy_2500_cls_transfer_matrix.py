#!/usr/bin/env python3
"""Plot the three-seed PRODIGY 2.5k downstream classification transfer matrix."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT_PATH = HERE / "data/classification_ladder/classification_long.tsv"
OUTPUT_DATA = HERE / "data/prodigy_2500_cls_transfer_matrix_with_mixture.csv"
OUTPUT_PNG = HERE / "figures/pngs/prodigy_2500_cls_transfer_matrix_with_mixture.png"
OUTPUT_PDF = HERE / "figures/pdfs/prodigy_2500_cls_transfer_matrix_with_mixture.pdf"

MODELS = ["ss_ukr_rus", "ss_ukr_rus_suspended", "ss_twibot20", "all9"]
TARGETS = [
    "covid_political",
    "election2020",
    "facebook_page_reference",
    "twibot20",
    "ukr_rus_suspended",
]
LABELS = {
    "ss_ukr_rus": "UKR/RUS",
    "ss_ukr_rus_suspended": "UKR/RUS suspended",
    "ss_twibot20": "TwiBot-20",
    "all9": "All-nine mixture",
    "covid_political": "COVID political",
    "election2020": "Election 2020",
    "facebook_page_reference": "Facebook pages",
    "twibot20": "TwiBot-20",
    "ukr_rus_suspended": "UKR/RUS suspended",
}


def build_matrix() -> pd.DataFrame:
    raw = pd.read_csv(INPUT_PATH, sep="\t")
    selected = raw[raw["model_id"].isin(MODELS)]
    seed_counts = selected.groupby(["model_id", "dataset"])["training_seed"].nunique()
    expected = pd.MultiIndex.from_product([MODELS, TARGETS], names=["model_id", "dataset"])
    if not seed_counts.reindex(expected).eq(3).all():
        raise ValueError("incomplete three-seed 2.5k classification matrix")
    matrix = (
        selected.groupby(["model_id", "dataset"])["roc_auc"]
        .mean()
        .unstack("dataset")
        .reindex(index=MODELS, columns=TARGETS)
    )
    matrix["min"] = matrix[TARGETS].min(axis=1)
    matrix["mean"] = matrix[TARGETS].mean(axis=1)
    matrix["max"] = matrix[TARGETS].max(axis=1)
    return matrix


def plot(matrix: pd.DataFrame) -> None:
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(13.7, 5.8))
    image = ax.imshow(values, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            color = "white" if value < 0.72 else "#161616"
            ax.text(col, row, f"{value:.3f}", ha="center", va="center", color=color, fontsize=9)

    ax.set_xticks(range(len(matrix.columns)), [LABELS.get(column, column.title()) for column in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), [LABELS[index] for index in matrix.index])
    ax.tick_params(axis="x", rotation=38, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlabel("Downstream classification target")
    ax.set_ylabel("Pretraining source or mixture")
    ax.axhline(len(MODELS) - 1.5, color="white", linewidth=3.0)
    ax.axvline(len(TARGETS) - 0.5, color="white", linewidth=3.0)
    ax.set_title("PRODIGY 2.5k classification transfer matrix", loc="left", fontsize=15, fontweight="bold", pad=42)
    ax.text(
        0,
        1.055,
        "Three-seed mean ROC-AUC · available single-source checkpoints plus the all-nine mixture · summaries span five targets",
        transform=ax.transAxes,
        color="#77746e",
        fontsize=9,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.018)
    colorbar.set_label("Classification ROC-AUC")
    fig.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    matrix = build_matrix()
    matrix.rename_axis("model_source").to_csv(OUTPUT_DATA)
    plot(matrix)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
