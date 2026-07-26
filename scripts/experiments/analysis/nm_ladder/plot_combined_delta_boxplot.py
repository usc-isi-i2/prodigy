#!/usr/bin/env python3
"""Combined delta boxplot: the 8 single-source specialists (sorted by MEAN gap, then
flipped so worst->best generalist runs left->right, up to the divider) followed by the
8 ladder models in entry order (ukr, +cov, ... , +cphk = all 8).

Style: uncoloured neutral boxes; one dot per test graph coloured by that graph, no dot
borders; y inverted (0 = best at top). All 16 models share one frontier ('best' = best
of all 16 on each test graph), so specialist and ladder columns are directly comparable.
Ladder columns are labelled by the graph that ENTERS at that rung.

Writes figures/boxplots/nm_ladder_plus_ss_delta_boxplot.pdf (vector) + .png (300 dpi).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures" / "boxplots"
OUT.mkdir(parents=True, exist_ok=True)

# Legible abbreviations, used as the canonical keys/labels everywhere.
GRAPHS = ["ukr", "cov", "mid", "cpol", "elec", "ususp", "twi", "cphk"]
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73",
             "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
COLOR = {g: OKABE_ITO[i] for i, g in enumerate(GRAPHS)}

# AUCs in GRAPHS column order: ukr, cov, mid, cpol, elec, ususp, twi, cphk.
SINGLE = [                                   # model = the one graph trained on
    ("cov",   [.9264, .9805, .8842, .8498, .8345, .7861, .9258, .7196]),
    ("ukr",   [.9470, .9730, .8811, .8394, .8262, .7894, .9218, .7140]),
    ("mid",   [.7970, .8790, .9252, .8353, .8047, .6440, .8613, .6256]),
    ("cpol",  [.6301, .6985, .6878, .9145, .7833, .5510, .7374, .5444]),
    ("elec",  [.6018, .6551, .6804, .7872, .9519, .5627, .7096, .5475]),
    ("ususp", [.7695, .8307, .7247, .7325, .7275, .9640, .7647, .6225]),
    ("twi",   [.8688, .9473, .8600, .8431, .8025, .7115, .9487, .6895]),
    ("cphk",  [.6808, .7579, .7629, .6827, .6412, .6025, .7378, .9055]),
]
LADDER = [                                   # label = graph that enters at this rung
    ("ukr",    [.9480, .9730, .8740, .8490, .8280, .7710, .9210, .7240]),
    ("+cov",   [.9450, .9800, .8850, .8430, .8280, .7750, .9250, .7260]),
    ("+mid",   [.9410, .9780, .9150, .8300, .8150, .7770, .9270, .7200]),
    ("+cpol",  [.9344, .9753, .9093, .9113, .8297, .7768, .9234, .7235]),
    ("+elec",  [.9346, .9754, .9086, .9102, .9259, .7693, .9254, .7261]),
    ("+ususp", [.9325, .9744, .9073, .9106, .9241, .9340, .9242, .7239]),
    ("+twi",   [.9321, .9748, .9033, .9076, .9198, .9256, .9377, .7267]),
    ("+cphk",  [.9340, .9750, .9080, .9060, .9200, .9310, .9370, .8670]),
]

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "figure.dpi": 130,
})


def main() -> int:
    A = np.array([m[1] for m in SINGLE + LADDER])   # 16 x 8
    best = A.max(axis=0)                              # frontier over all 16, per graph

    def gaps(row):
        return [best[j] - row[j] for j in range(8)]

    # sort single by mean gap, then FLIP -> worst generalist left, best next to divider
    single_ordered = sorted(SINGLE, key=lambda m: float(np.mean(gaps(m[1]))), reverse=True)
    n_s = len(single_ordered)
    models = single_ordered + LADDER
    positions = list(range(n_s)) + [n_s + 1 + i for i in range(len(LADDER))]  # blank gap

    fig, ax = plt.subplots(figsize=(14.5, 5.8))
    rng = np.random.default_rng(0)
    for pos, m in zip(positions, models):
        data = gaps(m[1])
        ax.boxplot([data], positions=[pos], widths=0.6, patch_artist=False,
                   medianprops=dict(color="black", lw=1.6),
                   boxprops=dict(color="0.35", lw=1.2),
                   whiskerprops=dict(color="0.45", lw=1.0),
                   capprops=dict(color="0.45", lw=1.0),
                   flierprops=dict(marker=""), zorder=2)
        jitter = rng.uniform(-0.17, 0.17, size=8)
        ax.scatter(pos + jitter, data, c=[COLOR[g] for g in GRAPHS], marker="o",
                   s=44, edgecolors="none", zorder=3)          # no dot border

    ax.axvline(n_s, color="0.6", lw=1.0, ls="--", zorder=1)    # group divider
    ax.set_xticks(positions)
    ax.set_xticklabels([m[0] for m in models], rotation=25, ha="right")
    ax.set_xlim(-0.7, positions[-1] + 0.7)
    ax.grid(axis="x", visible=False)

    top = float((best[None, :] - A).max())
    ax.set_ylabel("AUC gap to the best model on that graph  (0 = best)")
    ax.set_ylim(top * 1.05, -top * 0.03)                      # invert: best at top
    ax.set_title("How far each model trails the best on each test graph — specialists vs ladder\n"
                 "(best over all 16 models; single-source by mean gap worst→best; ladder in entry order; 1 seed)",
                 fontsize=11, pad=26)

    tb = ax.get_xaxis_transform()  # x = data, y = axes fraction
    ax.text((n_s - 1) / 2, 1.005, "single-source specialists", transform=tb,
            ha="center", va="bottom", fontsize=10.5, color="0.3")
    ax.text(n_s + 1 + (len(LADDER) - 1) / 2, 1.005, "merged ladder (label = graph added)",
            transform=tb, ha="center", va="bottom", fontsize=10.5, color="0.3")

    handles = [Line2D([0], [0], linestyle="none", marker="o", markerfacecolor=COLOR[g],
                      markeredgecolor="none", markersize=8, label=g) for g in GRAPHS]
    ax.legend(handles=handles, title="test graph", bbox_to_anchor=(1.005, 1.0),
              loc="upper left", frameon=False, fontsize=9, handletextpad=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "nm_ladder_plus_ss_delta_boxplot.pdf", bbox_inches="tight")
    fig.savefig(OUT / "nm_ladder_plus_ss_delta_boxplot.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote nm_ladder_plus_ss_delta_boxplot.(pdf|png @300dpi) to {OUT}")
    print("single order (worst→best generalist):", [m[0] for m in single_ordered])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
