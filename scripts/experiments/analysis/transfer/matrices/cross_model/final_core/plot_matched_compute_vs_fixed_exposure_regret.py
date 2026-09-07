#!/usr/bin/env python3
"""Compare matched-compute and fixed-exposure classification regret ladders."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
MATCHED = HERE / "data/classification_ladder/classification_long.tsv"
FIXED = (
    HERE.parents[2]
    / "ablations/prodigy_nm/downstream/nm_ladder_downstream_nhop2/data/downstream_long.csv"
)
PNG = HERE / "figures/pngs/prodigy_matched_compute_vs_fixed_exposure_regret.png"
PDF = HERE / "figures/pdfs/prodigy_matched_compute_vs_fixed_exposure_regret.pdf"
ORDERS = ("A", "C")
COLORS = {"A": "#0072B2", "C": "#009E73"}
MATCHED_RUNG1 = {"A": "ss_ukr_rus", "C": "ss_twibot20"}


def matched_model(order: str, rung: int) -> str:
    if rung == 1:
        return MATCHED_RUNG1[order]
    if rung == 9:
        return "all9"
    return f"ord{order}_r{rung}"


def load_matched() -> dict[str, list[tuple[int, float, float]]]:
    seeds: dict[tuple[str, str], list[float]] = defaultdict(list)
    sources: dict[str, set[str]] = {}
    with MATCHED.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            model, target = row["model_id"], row["dataset"]
            seeds[(model, target)].append(float(row["roc_auc"]))
            sources[model] = set(row["sources"].split(","))
    means = {key: float(np.mean(values)) for key, values in seeds.items()}
    targets = sorted({target for _, target in means})
    best = {target: max(means[(model, target)] for model in sources) for target in targets}
    points = {order: [] for order in ORDERS}
    for order in ORDERS:
        for rung in range(1, 10):
            model = matched_model(order, rung)
            home = [target for target in targets if target in sources[model]]
            if not home:
                continue
            x = 100 * np.mean([means[(model, t)] - best[t] for t in home])
            y = 100 * np.mean([means[(model, t)] - best[t] for t in targets])
            points[order].append((rung, float(x), float(y)))
    return points


def load_fixed() -> dict[str, list[tuple[int, float, float]]]:
    values: dict[tuple[str, int, str], float] = {}
    home: dict[tuple[str, int], set[str]] = defaultdict(set)
    with FIXED.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not (row["variant"] == "fixed10k" and row["task"] == "classification"
                    and row["metric"] == "roc_auc" and row["primary"] == "1"
                    and row["order"] in ORDERS):
                continue
            key = (row["order"], int(row["rung"]), row["dataset"])
            values[key] = float(row["value"])
            if row["in_training"] == "1":
                home[(key[0], key[1])].add(key[2])
    targets = sorted({target for _, _, target in values})
    best = {target: max(value for (*_, name), value in values.items() if name == target)
            for target in targets}
    points = {order: [] for order in ORDERS}
    for order in ORDERS:
        for rung in range(1, 9):
            home_targets = sorted(home.get((order, rung), set()))
            if not home_targets:
                continue
            x = 100 * np.mean([values[(order, rung, t)] - best[t] for t in home_targets])
            y = 100 * np.mean([values[(order, rung, t)] - best[t] for t in targets])
            points[order].append((rung, float(x), float(y)))
    return points


def main() -> None:
    protocols = (("Matched compute · 2.5k steps", load_matched(), 9, "5 targets · 3 seeds"),
                 ("Fixed exposure · 10k/source", load_fixed(), 8, "4 targets · 1 seed"))
    all_points = [point for _, tables, _, _ in protocols for points in tables.values() for point in points]
    all_x, all_y = [p[1] for p in all_points], [p[2] for p in all_points]
    center_x, center_y = (min(all_x) + max(all_x)) / 2, (min(all_y) + max(all_y)) / 2
    half = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) * 1.18 / 2
    xlim, ylim = (center_x - half, center_x + half), (center_y - half, center_y + half)
    diagonal = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 10.0), sharex=True, sharey=True)
    for row, (protocol, tables, final_rung, note) in enumerate(protocols):
        for col, order in enumerate(ORDERS):
            ax = axes[row, col]
            points = tables[order]
            xs, ys = [p[1] for p in points], [p[2] for p in points]
            color = COLORS[order]
            ax.plot(diagonal, diagonal, "--", color="#9a988e", lw=1.2, zorder=1)
            ax.axvline(0, ls=":", color="#c7c5bc", lw=1.0, zorder=1)
            ax.plot(xs, ys, "-", color=color, lw=1.8, alpha=.58, zorder=2)
            for rung, x, y in points:
                final = rung == final_rung
                ax.scatter(x, y, s=170 if final else 78, color=color,
                           edgecolor="white" if final else "none", linewidth=1.4, zorder=4)
                ax.annotate(f"all-{final_rung}" if final else f"L{rung}", (x, y),
                            xytext=(-8, 10) if final else (5, 5), textcoords="offset points",
                            fontsize=8.5 if final else 7.5,
                            color="#1b3a09" if final else color,
                            fontweight="bold" if final else "normal",
                            ha="right" if final else "left", zorder=5)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{protocol} · Order {order}", fontsize=10.5, fontweight="bold")
            ax.text(.98, .03, note, transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.5, color="#666666")
            ax.tick_params(labelsize=8.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(True, color="#efeee9", lw=.6, zorder=0)
    for ax in axes[-1]:
        ax.set_xlabel("home-turf regret (AUC pts)", fontsize=9.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("overall regret (AUC pts)", fontsize=9.5)
    axes[0, -1].annotate("ideal (0,0)\nis up-right ↗", (xlim[1] - .1, ylim[1] - .1),
                         ha="right", va="top", fontsize=8.5, color="#8a887e")
    fig.suptitle("Classification regret: matched compute vs fixed exposure", fontsize=13.5, y=.995)
    fig.text(.5, .006, "Regret is target-wise and normalized within each protocol's evaluation panel.",
             ha="center", fontsize=8, color="#666666")
    fig.tight_layout(rect=(0, .025, 1, .97), w_pad=1.2, h_pad=1.4)

    PNG.parent.mkdir(parents=True, exist_ok=True)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=300, bbox_inches="tight")
    fig.savefig(PDF, bbox_inches="tight")
    print(PNG)
    print(PDF)


if __name__ == "__main__":
    main()
