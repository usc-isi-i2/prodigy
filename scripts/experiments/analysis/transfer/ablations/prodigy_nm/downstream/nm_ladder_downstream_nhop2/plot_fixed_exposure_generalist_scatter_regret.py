#!/usr/bin/env python3
"""Plot fixed-exposure classification ladders in home/overall regret space."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
INPUT = HERE / "data/downstream_long.csv"
PNG = HERE / "figures/fixed_exposure_generalist_scatter_regret_ladder.png"
PDF = HERE / "figures/fixed_exposure_generalist_scatter_regret_ladder.pdf"
ORDERS = ("A", "C")
COLORS = {"A": "#0072B2", "C": "#009E73"}


def load_values() -> tuple[dict[tuple[str, int, str], float], dict[tuple[str, int], set[str]]]:
    values: dict[tuple[str, int, str], float] = {}
    home: dict[tuple[str, int], set[str]] = {}
    with INPUT.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not (
                row["variant"] == "fixed10k"
                and row["task"] == "classification"
                and row["metric"] == "roc_auc"
                and row["primary"] == "1"
                and row["order"] in ORDERS
            ):
                continue
            order, rung, target = row["order"], int(row["rung"]), row["dataset"]
            key = (order, rung, target)
            if key in values:
                raise ValueError(f"duplicate cell: {key}")
            values[key] = float(row["value"])
            if row["in_training"] == "1":
                home.setdefault((order, rung), set()).add(target)
    return values, home


def main() -> None:
    values, home = load_values()
    targets = sorted({target for _, _, target in values})
    best = {target: max(value for (order, rung, name), value in values.items() if name == target)
            for target in targets}

    points: dict[str, list[tuple[int, float, float]]] = {order: [] for order in ORDERS}
    for order in ORDERS:
        for rung in range(1, 9):
            home_targets = sorted(home.get((order, rung), set()))
            if not home_targets:
                continue
            home_regret = 100 * np.mean([values[(order, rung, t)] - best[t] for t in home_targets])
            overall_regret = 100 * np.mean([values[(order, rung, t)] - best[t] for t in targets])
            points[order].append((rung, float(home_regret), float(overall_regret)))

    all_x = [x for order_points in points.values() for _, x, _ in order_points]
    all_y = [y for order_points in points.values() for _, _, y in order_points]
    center_x, center_y = (min(all_x) + max(all_x)) / 2, (min(all_y) + max(all_y)) / 2
    half = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) * 1.18 / 2
    xlim, ylim = (center_x - half, center_x + half), (center_y - half, center_y + half)
    diagonal = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.4), sharex=True, sharey=True)
    for ax, order in zip(axes, ORDERS):
        order_points = points[order]
        xs, ys = [p[1] for p in order_points], [p[2] for p in order_points]
        color = COLORS[order]
        ax.plot(diagonal, diagonal, "--", color="#9a988e", lw=1.2, zorder=1)
        ax.axvline(0, ls=":", color="#c7c5bc", lw=1.0, zorder=1)
        ax.plot(xs, ys, "-", color=color, lw=1.8, alpha=.58, zorder=2)
        for rung, x, y in order_points:
            is_all8 = rung == 8
            ax.scatter(x, y, s=180 if is_all8 else 84, color=color,
                       edgecolor="white" if is_all8 else "none", linewidth=1.5, zorder=4)
            ax.annotate("all-8" if is_all8 else f"L{rung}", (x, y),
                        xytext=(-8, 10) if is_all8 else (5, 5), textcoords="offset points",
                        fontsize=9 if is_all8 else 8, color="#1b3a09" if is_all8 else color,
                        fontweight="bold" if is_all8 else "normal",
                        ha="right" if is_all8 else "left", zorder=5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Order {order}", fontsize=11, fontweight="bold")
        ax.set_xlabel("home-turf regret (AUC pts)", fontsize=9.5)
        ax.tick_params(labelsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color="#efeee9", lw=.6, zorder=0)
    axes[0].set_ylabel(f"overall regret over {len(targets)} targets (AUC pts)", fontsize=9.5)
    axes[-1].annotate("ideal (0,0)\nis up-right ↗", (xlim[1] - .1, ylim[1] - .1),
                      ha="right", va="top", fontsize=8.5, color="#8a887e")
    fig.suptitle("Fixed exposure: classification regret (10k updates/source)", fontsize=13, y=.98)
    fig.text(.99, .012, "1 training seed", ha="right", fontsize=8, color="#666666")
    fig.tight_layout(rect=(0, 0, 1, .94), w_pad=1.2)

    PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=300, bbox_inches="tight")
    fig.savefig(PDF, bbox_inches="tight")
    print(PNG)
    print(PDF)


if __name__ == "__main__":
    main()
