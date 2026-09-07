#!/usr/bin/env python3
"""Plot only the 2.5k fixed-compute ladders and the 1k saturation models."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import plot_nm_cls_auc_ladders as ladder


HERE = Path(__file__).resolve().parent
SATURATION = (
    HERE.parents[2]
    / "ablations/prodigy_nm/saturation/pretrain_saturation_nhop2/data/"
    "pretrain_saturation_nhop2_long.csv"
)
PNG = HERE / "figures/pngs/low_compute_1k_2p5k_results.png"
PDF = HERE / "figures/pdfs/low_compute_1k_2p5k_results.pdf"

ARM_COLORS = {"all8": "#7B3294", "covid": "#008837", "ukr": "#C51B7D"}
ARM_LABELS = {"all8": "All 8 graphs", "covid": "COVID graph", "ukr": "UKR graph"}


def main() -> None:
    nm = ladder.read_metric(ladder.NM_PATH, "roc_auc_ovr_macro_logged")
    cls = ladder.read_metric(ladder.CLS_PATH, "roc_auc")
    sources = ladder.read_sources(ladder.CLS_PATH)
    saturation = pd.read_csv(SATURATION)
    saturation = saturation[
        saturation.step.eq(1000)
        & saturation.task.eq("classification")
        & saturation.target.isna()
    ].copy()

    rungs = np.arange(1, 10)
    fig, axes = plt.subplots(3, 5, figsize=(16.2, 9.1), sharey="row")

    for column, target in enumerate(ladder.TARGETS):
        axes[0, column].set_title(ladder.TITLES[target], fontsize=10.5, fontweight="bold")
        for order in ladder.ORDERS:
            models = [ladder.model_for(order, rung) for rung in rungs]
            entered = ladder.entry_rung(sources, order, target)
            relative_rungs = rungs - entered
            color = ladder.ORDER_COLORS[order]
            for row, metric in ((0, nm), (1, cls)):
                values = np.array(
                    [[metric[(seed, model, target)] for model in models] for seed in range(3)]
                )
                ax = axes[row, column]
                ax.fill_between(
                    relative_rungs, values.min(0), values.max(0),
                    color=color, alpha=.09, linewidth=0,
                )
                ax.plot(relative_rungs, values.mean(0), color=color, lw=1.9, marker="o", ms=3)

        for row in (0, 1):
            ax = axes[row, column]
            ax.set_xlim(-8.35, .35)
            ax.set_xticks(np.arange(-8, 1, 2))
            ax.axvline(0, color="#777777", lw=1.15, ls=":", zorder=0)
            ax.grid(axis="y", color="#d9d9d9", linewidth=.7)
            ax.spines[["top", "right"]].set_visible(False)
        axes[1, column].set_xlabel("rungs before target entry")

        ax = axes[2, column]
        part = saturation[saturation.dataset.eq(target)]
        if part.empty:
            ax.text(.5, .5, "not evaluated", ha="center", va="center", color="#777777")
            ax.set_xticks([])
        else:
            arms = [arm for arm in ("all8", "covid", "ukr") if arm in set(part.arm)]
            for x, arm in enumerate(arms):
                value = float(part.loc[part.arm.eq(arm), "value"].iloc[0])
                ax.scatter(x, value, s=48, color=ARM_COLORS[arm], zorder=3)
            ax.set_xticks(range(len(arms)), [ARM_LABELS[a].replace(" graph", "") for a in arms], rotation=25, ha="right")
        ax.grid(axis="y", color="#d9d9d9", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0, 0].set_ylabel("NM AUC")
    axes[1, 0].set_ylabel("Classification AUC")
    axes[2, 0].set_ylabel("Classification AUC")
    fig.text(.008, .905, "a  2.5k fixed-compute models", fontweight="bold")
    fig.text(.008, .318, "b  1k single-source / all-source models", fontweight="bold")

    handles = [Line2D([0], [0], color=ladder.ORDER_COLORS[o], lw=2.3, label=f"Order {o}") for o in ladder.ORDERS]
    handles.append(Line2D([0], [0], color="#777777", lw=1.2, ls=":", label="target enters mixture"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.005))
    fig.suptitle("Low-compute downstream results (no 10k/40k models)", y=.995, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, .055, 1, .96), w_pad=1.0, h_pad=2.0)
    PNG.parent.mkdir(parents=True, exist_ok=True)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=220, bbox_inches="tight")
    fig.savefig(PDF, bbox_inches="tight")
    plt.close(fig)
    print(PNG)


if __name__ == "__main__":
    main()
