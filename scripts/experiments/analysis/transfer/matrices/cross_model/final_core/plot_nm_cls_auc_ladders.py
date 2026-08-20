#!/usr/bin/env python3
"""Plot mean NM and downstream classification AUC along the final-core ladders."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
NM_PATH = HERE / "data/prodigy_final_core/log_recovered_metrics/physical_metrics.tsv"
CLS_PATH = HERE / "data/classification_ladder/classification_long.tsv"
PDF = HERE / "figures/pdfs/prodigy_nm_vs_cls_auc_ladders.pdf"
PNG = HERE / "figures/pngs/prodigy_nm_vs_cls_auc_ladders.png"

TARGETS = (
    "covid_political",
    "election2020",
    "facebook_page_reference",
    "twibot20",
    "ukr_rus_suspended",
)
TITLES = {
    "covid_political": "COVID political",
    "election2020": "Election 2020",
    "facebook_page_reference": "Facebook pages",
    "twibot20": "TwiBot-20",
    "ukr_rus_suspended": "UKR–RUS suspended",
}
ORDERS = "ABC"
ORDER_COLORS = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73"}
RUNG1 = {"A": "ss_ukr_rus", "B": "ss_ukr_rus_suspended", "C": "ss_twibot20"}


def model_for(order: str, rung: int) -> str:
    if rung == 1:
        return RUNG1[order]
    if rung == 9:
        return "all9"
    return f"ord{order}_r{rung}"


def read_metric(path: Path, metric: str) -> dict[tuple[int, str, str], float]:
    values: dict[tuple[int, str, str], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            seed_key = "training_seed" if "training_seed" in row else "seed"
            target_key = "dataset" if "dataset" in row else "target"
            model_key = "model_id" if "model_id" in row else "model"
            key = (int(row[seed_key]), row[model_key], row[target_key])
            values[key] = float(row[metric])
    return values


def read_sources(path: Path) -> dict[str, set[str]]:
    """Return the recorded pretraining mixture for each physical model."""
    sources: dict[str, set[str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            model_sources = set(row["sources"].split(","))
            previous = sources.setdefault(row["model_id"], model_sources)
            if previous != model_sources:
                raise ValueError(f"Inconsistent sources for {row['model_id']}")
    return sources


def entry_rung(sources: dict[str, set[str]], order: str, target: str) -> int:
    """Find the first rung whose recorded mixture contains the target graph."""
    for rung in range(1, 10):
        if target in sources[model_for(order, rung)]:
            return rung
    raise ValueError(f"{target} never enters order {order}")


def main() -> None:
    nm = read_metric(NM_PATH, "roc_auc_ovr_macro_logged")
    cls = read_metric(CLS_PATH, "roc_auc")
    sources = read_sources(CLS_PATH)
    rungs = np.arange(1, 10)
    fig, axes = plt.subplots(2, 5, figsize=(16.2, 6.4), sharex=True, sharey=True)

    for column, target in enumerate(TARGETS):
        axes[0, column].set_title(TITLES[target], fontsize=10.5, fontweight="bold")
        for order in ORDERS:
            model_ids = [model_for(order, rung) for rung in rungs]
            nm_seed = np.array([[nm[(seed, model, target)] for model in model_ids] for seed in range(3)])
            cls_seed = np.array([[cls[(seed, model, target)] for model in model_ids] for seed in range(3)])
            color = ORDER_COLORS[order]
            entered = entry_rung(sources, order, target)
            relative_rungs = rungs - entered
            for row, values, line_style, marker in (
                (0, nm_seed, "-", "o"),
                (1, cls_seed, "--", "s"),
            ):
                ax = axes[row, column]
                ax.fill_between(
                    relative_rungs, values.min(0), values.max(0),
                    color=color, alpha=.09, linewidth=0,
                )
                ax.plot(
                    relative_rungs, values.mean(0), color=color, lw=1.9,
                    ls=line_style, marker=marker, ms=3.0,
                )

        for row in range(2):
            ax = axes[row, column]
            ax.set_xlim(-8.35, .35)
            ax.set_xticks(np.arange(-8, 1, 2))
            ax.axvline(0, color="#777777", lw=1.15, ls=":", zorder=0)
            ax.grid(axis="y", color="#d9d9d9", linewidth=.7)
            ax.spines[["top", "right"]].set_visible(False)
        axes[1, column].set_xlabel("rungs relative to target entry")
    axes[0, 0].set_ylabel("NM AUC")
    axes[1, 0].set_ylabel("Classification AUC")

    handles = [Line2D([0], [0], color=ORDER_COLORS[o], lw=2.3, label=f"Order {o}") for o in ORDERS]
    handles += [
        Line2D([0], [0], color="#777777", lw=1.2, ls=":", label="target enters mix"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.01))
    fig.suptitle("PRODIGY: entry-aligned NM vs classification AUC", y=.99, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, .08, 1, .94), w_pad=1.0, h_pad=2.0)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF, bbox_inches="tight")
    fig.savefig(PNG, dpi=220, bbox_inches="tight")
    print(PDF)


if __name__ == "__main__":
    main()
