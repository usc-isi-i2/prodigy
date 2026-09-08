#!/usr/bin/env python3
"""Plot mean per-target NM AUC changes from each schedule's own first rung."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/prodigy-mpl-cache")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "data/nm_ladder_schedule_comparison_long.csv")
    parser.add_argument("--output", type=Path, default=HERE / "figures/schedule_mean_per_graph_delta.png")
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    cells = {(int(row["rung"]), row["test_graph"]): row for row in rows}
    rungs = sorted({rung for rung, _ in cells})
    graphs = sorted({graph for _, graph in cells})
    if len(cells) != len(rows) or rungs != list(range(1, 9)) or len(graphs) != 8:
        raise ValueError("Expected unique cells for eight rungs and eight target graphs")
    if set(cells) != {(rung, graph) for rung in rungs for graph in graphs}:
        raise ValueError("Every rung must contain the same eight target graphs")

    means = {}
    exported = []
    for method, column in (("Ours", "auc_interleaved"), ("Sequential", "auc_sequential")):
        values = np.array([[float(cells[rung, graph][column]) for graph in graphs] for rung in rungs])
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite AUC in {method}")
        deltas = values - values[0, :]
        means[method] = deltas.mean(axis=1)
        for i, rung in enumerate(rungs):
            for j, graph in enumerate(graphs):
                exported.append(dict(method=method, rung=rung, test_graph=graph,
                                     auc=values[i, j], baseline_auc=values[0, j], delta_auc=deltas[i, j]))

    # This is one fixed all-graph model repeated across x, not a merging ladder.
    # Its per-target change relative to itself is identically zero.
    means["Simple merging"] = np.zeros(len(rungs))
    assert all(values[0] == 0 for values in means.values())

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16,
                         "text.color": "#555555", "axes.labelcolor": "#666666",
                         "xtick.color": "#777777", "ytick.color": "#777777",
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(11.6, 5.3))
    fig.subplots_adjust(left=0.12, right=0.77, bottom=0.23, top=0.91)
    colors = {"Ours": "#387DFF", "Sequential": "#CB8577", "Simple merging": "#87A79A"}
    for method in ("Simple merging", "Sequential", "Ours"):
        ax.plot(rungs, means[method], color=colors[method], linewidth=2.4)
    ax.set(xlim=(0.95, 8.08), ylim=(-0.08, 0.08), xticks=rungs,
           yticks=np.arange(-0.08, 0.081, 0.04), xlabel="Number of training graphs",
           ylabel="Mean per-graph Δ NM ROC-AUC")
    ax.xaxis.labelpad = 14
    ax.yaxis.labelpad = 14
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: "0" if abs(y) < 1e-10 else f"{y:+.2f}"))
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[:].set_visible(False)
    ax.tick_params(axis="both", length=0, pad=9)
    for method, offset in (("Ours", 0), ("Simple merging", 18), ("Sequential", -19)):
        ax.annotate(method, (8, means[method][-1]), xytext=(18, offset),
                    textcoords="offset points", color=colors[method], fontsize=18,
                    va="center", annotation_clip=False)
    ax.annotate("Fixed all-graph baseline", (8, 0), xytext=(18, 0),
                textcoords="offset points", fontsize=11, color="#888888",
                va="center", annotation_clip=False)
    fig.text(0.12, 0.065, "Higher is better · Each curve relative to its own starting scores", fontsize=12, color="#777777")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        path = args.output.with_suffix(suffix)
        fig.savefig(path, dpi=240, facecolor="white")
        print(path)
    plt.close(fig)
    data_path = HERE / "data/schedule_per_graph_deltas.csv"
    with data_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    for method, values in means.items():
        print(f"{method}: " + ", ".join(f"{v:+.6f}" for v in values))


if __name__ == "__main__":
    main()
