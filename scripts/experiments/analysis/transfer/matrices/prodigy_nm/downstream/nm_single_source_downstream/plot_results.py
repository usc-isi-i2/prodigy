#!/usr/bin/env python3
"""Plot classification and regression transfer for all eight single-source NM models."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np

SOURCE_ORDER = [
    "ukr_rus_twitter", "covid19_twitter", "twibot20", "midterm",
    "ukr_rus_suspended", "cp_hk_twitter", "covid_political", "election2020",
]
SHORT = {
    "ukr_rus_twitter": "ukr", "covid19_twitter": "covid", "midterm": "midterm",
    "covid_political": "cov_pol", "election2020": "elec20",
    "ukr_rus_suspended": "ukr_susp", "twibot20": "twibot20",
    "cp_hk_twitter": "cp_hk",
}
CLASS_COLUMNS = [
    "covid_political", "election2020", "ukr_rus_suspended", "twibot20", "mean",
]
REG_COLUMNS = [
    "ukr_rus_twitter", "covid19_twitter", "midterm", "twibot20", "mean",
]


def read_by_source(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["source"]: row for row in csv.DictReader(handle)}


def text_color(value: float, low: float, high: float) -> str:
    fraction = (value - low) / (high - low)
    return "white" if fraction < 0.20 or fraction > 0.82 else "black"


def draw_heatmap(ax, data, columns, title, cmap, norm, colorbar_label):
    image = ax.imshow(data, cmap=cmap, norm=norm, aspect="equal")
    labels = [SHORT.get(column, column) for column in columns[:-1]] + ["mean"]
    ax.set_xticks(np.arange(len(columns)), labels, rotation=27, ha="right")
    ax.set_yticks(np.arange(len(SOURCE_ORDER)), [SHORT[source] for source in SOURCE_ORDER])
    ax.set_xlabel("evaluation graph")
    ax.set_ylabel("NM pretraining source")
    ax.set_title(title, fontsize=11)
    ax.axvline(len(columns) - 1.5, color="black", linewidth=2)
    for i, source in enumerate(SOURCE_ORDER):
        for j, column in enumerate(columns):
            value = data[i, j]
            if isinstance(norm, TwoSlopeNorm):
                color = "white" if abs(value) > 0.17 else "black"
            else:
                color = text_color(value, norm.vmin, norm.vmax)
            is_diagonal = column != "mean" and source == column
            ax.text(
                j, i, f"{value:.3f}", ha="center", va="center", color=color,
                fontsize=9, fontweight=("bold" if is_diagonal else "normal"),
            )
            if is_diagonal:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False,
                    edgecolor="black", linewidth=2,
                ))
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(SOURCE_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9, alpha=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    colorbar.set_label(colorbar_label)


def main() -> int:
    here = Path(__file__).resolve().parent
    class_rows = read_by_source(here / "data/classification.csv")
    reg_rows = read_by_source(here / "data/regression_by_dataset.csv")
    class_data = np.array([
        [float(class_rows[source][column]) for column in CLASS_COLUMNS]
        for source in SOURCE_ORDER
    ])
    reg_data = np.array([
        [float(reg_rows[source][column]) for column in REG_COLUMNS]
        for source in SOURCE_ORDER
    ])

    plt.rcParams.update({"font.size": 10, "figure.dpi": 140})
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.9), constrained_layout=True)
    draw_heatmap(
        axes[0], class_data, CLASS_COLUMNS,
        "Node classification\n10-shot ROC-AUC",
        "RdYlGn", Normalize(0.5, 1.0), "ROC-AUC",
    )
    draw_heatmap(
        axes[1], reg_data, REG_COLUMNS,
        "Node regression\n10-shot Spearman, mean over 3 profile targets",
        "RdYlGn", TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=0.25), "Spearman",
    )
    fig.suptitle(
        "Downstream transfer of eight single-source NM encoders\n"
        "rows follow NM donor-strength order · outlined cells are in-domain",
        fontsize=13,
    )
    out_dir = here / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"single_source_downstream_heatmaps.{suffix}", dpi=180)
    plt.close(fig)
    print(f"wrote {out_dir}/single_source_downstream_heatmaps.(png|pdf)")

    class_fig, class_ax = plt.subplots(figsize=(7.2, 7.4), constrained_layout=True)
    draw_heatmap(
        class_ax, class_data, CLASS_COLUMNS,
        "10-shot ROC-AUC",
        "RdYlGn", Normalize(0.5, 1.0), "ROC-AUC",
    )
    class_fig.suptitle(
        "Node classification transfer of eight single-source NM encoders\n"
        "rows follow NM donor-strength order · outlined cells are in-domain",
        fontsize=13,
    )
    for suffix in ("png", "pdf"):
        class_fig.savefig(
            out_dir / f"single_source_classification_heatmap.{suffix}", dpi=180
        )
    plt.close(class_fig)
    print(f"wrote {out_dir}/single_source_classification_heatmap.(png|pdf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
