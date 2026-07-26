#!/usr/bin/env python3
"""Ladder analogue of nmss_delta_boxplot: x = ladder model (L1..all8), each column a
boxplot of that model's AUC gap to the best LADDER model on each of the 8 test graphs
(0 = best), y inverted so 'best' sits at the top.

Differences from the single-source version (per request):
  - each TEST GRAPH gets a colour; points are all CIRCLES (colour is the only encoding)
  - the boxplots are uncoloured (neutral grey line boxes, no fills)

Ladder kept in its natural order L1 -> all8 (training set grows), so the boxes visibly
tighten and rise toward the frontier as sources are added. "Best" is taken over the 8
ladder models only (self-contained, like the nmss figure).

Writes nm_ladder_delta_boxplot.(pdf|png) into ./figures/. matplotlib + numpy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

# Test-graph order + short labels + colours (Okabe-Ito, matches the nmss figures).
GRAPHS = ["ukr", "covid", "midterm", "cov_pol", "elec20", "ukr_susp", "twibot20", "cp_hk"]
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73",
             "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
COLOR = {g: OKABE_ITO[i] for i, g in enumerate(GRAPHS)}

# 8 ladder models; AUCs in GRAPHS column order (from nm_ladder_full.csv).
LADDER = [
    ("L1 · ukr",       [.9480, .9730, .8740, .8490, .8280, .7710, .9210, .7240]),
    ("L2 · +cov",      [.9450, .9800, .8850, .8430, .8280, .7750, .9250, .7260]),
    ("L3 · +mid",      [.9410, .9780, .9150, .8300, .8150, .7770, .9270, .7200]),
    ("L4 · +cov_pol",  [.9344, .9753, .9093, .9113, .8297, .7768, .9234, .7235]),
    ("L5 · +elec20",   [.9346, .9754, .9086, .9102, .9259, .7693, .9254, .7261]),
    ("L6 · +ukr_susp", [.9325, .9744, .9073, .9106, .9241, .9340, .9242, .7239]),
    ("L7 · +twibot20", [.9321, .9748, .9033, .9076, .9198, .9256, .9377, .7267]),
    ("L8 · all8",      [.9340, .9750, .9080, .9060, .9200, .9310, .9370, .8670]),
]

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "figure.dpi": 130,
})


def main() -> int:
    labels = [m[0] for m in LADDER]
    A = np.array([m[1] for m in LADDER])          # 8 models x 8 test graphs
    best = A.max(axis=0)                            # best ladder model per test graph
    gaps = best[None, :] - A                        # >= 0, gap to best (per model, per graph)

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    rng = np.random.default_rng(0)
    for pos in range(len(LADDER)):
        data = gaps[pos]
        # uncoloured boxplot (neutral grey lines, no fill)
        ax.boxplot([data], positions=[pos], widths=0.6, patch_artist=False,
                   medianprops=dict(color="black", lw=1.6),
                   boxprops=dict(color="0.35", lw=1.2),
                   whiskerprops=dict(color="0.45", lw=1.0),
                   capprops=dict(color="0.45", lw=1.0),
                   flierprops=dict(marker=""), zorder=2)
        # one circle per test graph, coloured by the test graph
        jitter = rng.uniform(-0.17, 0.17, size=len(GRAPHS))
        ax.scatter(pos + jitter, data, c=[COLOR[g] for g in GRAPHS], marker="o", s=46,
                   edgecolor="0.25", linewidth=0.5, zorder=3)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel("ladder model  (training set grows →)")
    ax.grid(axis="x", visible=False)

    top = float(gaps.max())
    ax.set_ylabel("AUC gap to the best ladder model on that graph  (0 = best)")
    ax.set_ylim(top * 1.05, -top * 0.03)           # invert: 0 (best) at top
    ax.set_title("How far each ladder model trails the best on each test graph\n"
                 "(box tightens & rises toward 0 as sources are added; 1 seed)", fontsize=11)

    handles = [Line2D([0], [0], linestyle="none", marker="o", markerfacecolor=COLOR[g],
                      markeredgecolor="0.25", markersize=8, label=g) for g in GRAPHS]
    ax.legend(handles=handles, title="test graph", bbox_to_anchor=(1.01, 1.0),
              loc="upper left", frameon=False, fontsize=9, handletextpad=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"nm_ladder_delta_boxplot.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote nm_ladder_delta_boxplot.(pdf|png) to {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
