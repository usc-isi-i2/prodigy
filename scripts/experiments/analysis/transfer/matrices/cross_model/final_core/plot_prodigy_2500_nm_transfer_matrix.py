#!/usr/bin/env python3
"""Plot the three-seed PRODIGY 2.5k NM transfer matrix with the all9 mixture."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MATRIX_PATH = (
    HERE / "data/prodigy_final_core/auc/summary/"
    "single_source_roc_auc_ovr_macro_three_seed_mean.csv"
)
MIXTURE_PATH = HERE / "data/prodigy_final_core/log_recovered_metrics/physical_metrics.tsv"
OUTPUT_DATA = HERE / "data/prodigy_2500_nm_transfer_matrix_with_mixture.csv"
OUTPUT_PNG = HERE / "figures/pngs/prodigy_2500_nm_transfer_matrix_with_mixture.png"
OUTPUT_PDF = HERE / "figures/pdfs/prodigy_2500_nm_transfer_matrix_with_mixture.pdf"

TARGETS = [
    "ukr_rus", "covid", "midterm", "covid_political", "election2020",
    "ukr_rus_suspended", "twibot20", "cp_hk", "facebook_page_reference",
]
LABELS = {
    "ukr_rus": "UKR/RUS",
    "covid": "COVID-19",
    "midterm": "US midterm",
    "covid_political": "COVID political",
    "election2020": "Election 2020",
    "ukr_rus_suspended": "UKR/RUS suspended",
    "twibot20": "TwiBot-20",
    "cp_hk": "CP/HK",
    "facebook_page_reference": "Facebook pages",
    "all9_mixture": "All-nine mixture",
}


def build_matrix() -> pd.DataFrame:
    specialists = pd.read_csv(MATRIX_PATH).set_index("model_source").loc[TARGETS, TARGETS]
    raw = pd.read_csv(MIXTURE_PATH, sep="\t")
    mixture = (
        raw[raw["model_id"] == "all9"]
        .groupby("target")["roc_auc_ovr_macro_logged"]
        .mean()
        .reindex(TARGETS)
    )
    if mixture.isna().any() or specialists.isna().any().any():
        raise ValueError("incomplete 2.5k NM matrix or all9 mixture row")
    matrix = pd.concat([specialists, mixture.to_frame().T.rename(index={"roc_auc_ovr_macro_logged": "all9_mixture"})])
    matrix["min"] = matrix[TARGETS].min(axis=1)
    matrix["mean"] = matrix[TARGETS].mean(axis=1)
    matrix["max"] = matrix[TARGETS].max(axis=1)
    return matrix


def plot(matrix: pd.DataFrame) -> None:
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(17.2, 10.2))
    image = ax.imshow(values, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            color = "white" if value < 0.72 else "#161616"
            ax.text(col, row, f"{value:.3f}", ha="center", va="center", color=color, fontsize=8.5)

    ax.set_xticks(range(len(matrix.columns)), [LABELS.get(column, column.title()) for column in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), [LABELS[index] for index in matrix.index])
    ax.tick_params(axis="x", rotation=38, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlabel("NM evaluation graph")
    ax.set_ylabel("Pretraining source or mixture")
    ax.axhline(len(TARGETS) - 0.5, color="white", linewidth=3.0)
    ax.axvline(len(TARGETS) - 0.5, color="white", linewidth=3.0)
    ax.set_title("PRODIGY 2.5k NM transfer matrix", loc="left", fontsize=15, fontweight="bold", pad=42)
    ax.text(
        0, 1.025,
        "Three-seed mean ROC-AUC · final row is the all-nine mixture · summary columns span the nine evaluation graphs",
        transform=ax.transAxes, color="#77746e", fontsize=9,
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.018)
    colorbar.set_label("NM ROC-AUC")
    fig.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    matrix = build_matrix()
    OUTPUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    matrix.rename_axis("model_source").to_csv(OUTPUT_DATA)
    plot(matrix)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
