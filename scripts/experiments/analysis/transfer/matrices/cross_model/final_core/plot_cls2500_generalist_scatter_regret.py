#!/usr/bin/env python3
"""Plot the 2,500-step classification ladders in home/overall regret space."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
INPUT = HERE / "data/classification_ladder/classification_long.tsv"
PDF = HERE / "figures/pdfs/prodigy_cls2500_generalist_scatter_regret_ladder.pdf"
PNG = HERE / "figures/pngs/prodigy_cls2500_generalist_scatter_regret_ladder.png"

TARGETS = (
    "covid_political",
    "election2020",
    "facebook_page_reference",
    "twibot20",
    "ukr_rus_suspended",
)
ORDERS = "ABC"
COLORS = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73"}
RUNG1 = {"A": "ss_ukr_rus", "B": "ss_ukr_rus_suspended", "C": "ss_twibot20"}


def model_for(order: str, rung: int) -> str:
    if rung == 1:
        return RUNG1[order]
    if rung == 9:
        return "all9"
    return f"ord{order}_r{rung}"


def load_means() -> tuple[dict[tuple[str, str], float], dict[str, set[str]]]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    sources: dict[str, set[str]] = {}
    with INPUT.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            model, target = row["model_id"], row["dataset"]
            values[(model, target)].append(float(row["roc_auc"]))
            model_sources = set(row["sources"].split(","))
            if model in sources and sources[model] != model_sources:
                raise ValueError(f"inconsistent sources for {model}")
            sources[model] = model_sources
    means = {key: float(np.mean(seed_values)) for key, seed_values in values.items()}
    return means, sources


def main() -> None:
    means, sources = load_means()
    best = {target: max(means[(model, target)] for model in sources) for target in TARGETS}

    points: dict[str, list[tuple[int, float, float]]] = {order: [] for order in ORDERS}
    for order in ORDERS:
        for rung in range(1, 10):
            model = model_for(order, rung)
            home = [target for target in TARGETS if target in sources[model]]
            if not home:
                continue  # Order A rung 1 has no target in the five-target evaluation panel.
            home_regret = 100 * np.mean([means[(model, t)] - best[t] for t in home])
            overall_regret = 100 * np.mean([means[(model, t)] - best[t] for t in TARGETS])
            points[order].append((rung, float(home_regret), float(overall_regret)))

    all_x = [x for order_points in points.values() for _, x, _ in order_points]
    all_y = [y for order_points in points.values() for _, _, y in order_points]
    center_x = (min(all_x) + max(all_x)) / 2
    center_y = (min(all_y) + max(all_y)) / 2
    half = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) * 1.18 / 2
    xlim, ylim = (center_x - half, center_x + half), (center_y - half, center_y + half)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.4), sharex=True, sharey=True)
    diagonal = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
    for ax, order in zip(axes, ORDERS):
        order_points = points[order]
        xs = [point[1] for point in order_points]
        ys = [point[2] for point in order_points]
        color = COLORS[order]
        ax.plot(diagonal, diagonal, "--", color="#9a988e", lw=1.2, zorder=1)
        ax.axvline(0, ls=":", color="#c7c5bc", lw=1.0, zorder=1)
        ax.plot(xs, ys, "-", color=color, lw=1.8, alpha=.58, zorder=2)
        for rung, x, y in order_points:
            is_all9 = rung == 9
            ax.scatter(
                x, y, s=180 if is_all9 else 84, color=color,
                edgecolor="white" if is_all9 else "none", linewidth=1.5, zorder=4,
            )
            if not is_all9:
                ax.annotate(
                    f"L{rung}", (x, y), xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color=color, zorder=5,
                )
        all9 = order_points[-1]
        ax.annotate(
            "all-9", (all9[1], all9[2]), xytext=(-8, 10), textcoords="offset points",
            fontsize=9, color="#1b3a09", fontweight="bold", ha="right",
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Order {order}", fontsize=11, fontweight="bold")
        ax.set_xlabel("home-turf regret (AUC pts)", fontsize=9.5)
        ax.tick_params(labelsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color="#efeee9", lw=.6, zorder=0)
    axes[0].set_ylabel("overall regret over 5 targets (AUC pts)", fontsize=9.5)
    axes[-1].annotate(
        "ideal (0,0)\nis up-right ↗", (xlim[1] - .15, ylim[1] - .15),
        ha="right", va="top", fontsize=8.5, color="#8a887e",
    )
    fig.suptitle("Final-core ladders in regret space — classification (2.5k steps)",
                 fontsize=13, y=.98)
    fig.tight_layout(rect=(0, 0, 1, .94), w_pad=1.2)

    PDF.parent.mkdir(parents=True, exist_ok=True)
    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF, bbox_inches="tight")
    fig.savefig(PNG, dpi=300, bbox_inches="tight")
    print(PNG)
    print(PDF)


if __name__ == "__main__":
    main()
