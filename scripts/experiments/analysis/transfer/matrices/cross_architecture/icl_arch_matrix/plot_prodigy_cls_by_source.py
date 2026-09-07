#!/usr/bin/env python3
"""Plot downstream classification trajectories for each single-source PRODIGY model."""

from __future__ import annotations

import ast
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
OUTPUT = HERE / "figures" / "prodigy_native_objective_cls_by_source_900_seed0"
MEAN_OUTPUT = HERE / "figures" / "prodigy_native_objective_cls_by_target_900_seed0"

STEPS = (0, 100, 300, 900)
MEAN_STEPS = (0, 20, 60, 100, 300, 900)
SOURCES = (
    ("covid_political", "COVID political"),
    ("election2020", "Election 2020"),
    ("ukr_rus_suspended", "UKR/RUS suspended"),
    ("twibot20", "TwiBot-20"),
    ("facebook_page_reference", "Facebook pages"),
)
TARGETS = SOURCES
TARGET_LEGEND_LABELS = {
    "covid_political": "T-18",
    "election2020": "T-20",
}
TARGET_COLORS = ("#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377")


def load_rows() -> list[dict[str, str]]:
    with INPUT.open(encoding="utf-8", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["architecture"] == "prodigy"
        ]


def value(rows: list[dict[str, str]], source: str, target: str, step: int) -> float:
    if step == 0:
        matches = [
            row for row in rows
            if int(row["checkpoint_step"]) == 0 and row["dataset"] == target
        ]
    else:
        matches = [
            row for row in rows
            if int(row["checkpoint_step"]) == step
            and row["dataset"] == target
            and ast.literal_eval(row["sources"]) == [source]
        ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {source=}, {target=}, {step=}; got {len(matches)}")
    return float(matches[0]["roc_auc"])


def main() -> None:
    rows = load_rows()
    # log10(step + 1) preserves a meaningful zero checkpoint on a log-like axis.
    positions = np.log10(np.asarray(STEPS, dtype=float) + 1.0)
    curves = {
        (source, target): [value(rows, source, target, step) for step in STEPS]
        for source, _ in SOURCES
        for target, _ in TARGETS
    }

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), sharex=True, sharey=True)
    flat_axes = axes.ravel()

    for axis, (source, source_label) in zip(flat_axes, SOURCES):
        for (target, target_label), color in zip(TARGETS, TARGET_COLORS):
            axis.plot(
                positions,
                curves[(source, target)],
                color=color,
                marker="o",
                markersize=4.5,
                linewidth=2.1,
                label=target_label,
            )
        axis.set_title(f"Train: {source_label}")
        axis.set_xticks(positions, [str(step) for step in STEPS])
        axis.set_ylim(0.25, 1.0)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)

    flat_axes[-1].axis("off")
    flat_axes[-1].legend(
        *flat_axes[0].get_legend_handles_labels(),
        title="Classification target",
        loc="center",
        frameon=False,
        fontsize=11,
        title_fontsize=12,
    )

    for axis in axes[-1, :2]:
        axis.set_xlabel("Training checkpoint")
    for axis in axes[:, 0]:
        axis.set_ylabel("Classification ROC-AUC")

    fig.suptitle(
        "PRODIGY native NM pretraining: classification transfer by training graph",
        fontsize=19,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    mean_positions = np.arange(len(MEAN_STEPS))
    mean_fig, mean_axis = plt.subplots(figsize=(8.4, 5.2))
    for (target, target_label), color in zip(TARGETS, TARGET_COLORS):
        mean_values = [
            float(np.mean([value(rows, source, target, step) for source, _ in SOURCES]))
            for step in MEAN_STEPS
        ]
        mean_axis.plot(
            mean_positions,
            mean_values,
            color=color,
            marker="o",
            markersize=4.5,
            linewidth=2.2,
            label=TARGET_LEGEND_LABELS.get(target, target_label),
        )
    mean_axis.set_xticks(mean_positions, [str(step) for step in MEAN_STEPS])
    mean_axis.set_xlabel("Training checkpoint")
    mean_axis.set_ylabel("Classification ROC-AUC")
    mean_axis.set_ylim(0.25, 1.0)
    mean_axis.grid(axis="y", alpha=0.25)
    mean_axis.legend(frameon=False, ncol=2)
    mean_axis.set_title("PRODIGY native NM pretraining: target classification\n(mean over source models)")
    mean_fig.tight_layout()
    mean_fig.savefig(MEAN_OUTPUT.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    mean_fig.savefig(MEAN_OUTPUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(mean_fig)

    print(OUTPUT.with_suffix(".png"))
    print(OUTPUT.with_suffix(".pdf"))
    print(MEAN_OUTPUT.with_suffix(".png"))
    print(MEAN_OUTPUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
