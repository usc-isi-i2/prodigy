#!/usr/bin/env python3
"""Combined delta boxplot: the 8 single-source specialists (sorted by MEAN gap,
ascending) followed by the 8 ladder models (L1 -> all8, natural order), in one figure.

Same style as the two standalone delta boxplots: uncoloured neutral boxes, one circle
per test graph coloured by that graph, y inverted (0 = best at top). Because all 16
models share one y-axis, 'best' is the best of ALL 16 models on each test graph (one
common frontier), so specialist and ladder columns are directly comparable.

Writes figures/boxplots/nm_ladder_plus_ss_delta_boxplot.(pdf|png).
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

GRAPHS = ["ukr", "covid", "midterm", "cov_pol", "elec20", "ukr_susp", "twibot20", "cp_hk"]
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73",
             "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
COLOR = {g: OKABE_ITO[i] for i, g in enumerate(GRAPHS)}

# AUCs in GRAPHS column order.
SINGLE = [
    ("covid",    [.9264, .9805, .8842, .8498, .8345, .7861, .9258, .7196]),
    ("ukr",      [.9470, .9730, .8811, .8394, .8262, .7894, .9218, .7140]),
    ("midterm",  [.7970, .8790, .9252, .8353, .8047, .6440, .8613, .6256]),
    ("cov_pol",  [.6301, .6985, .6878, .9145, .7833, .5510, .7374, .5444]),
    ("elec20",   [.6018, .6551, .6804, .7872, .9519, .5627, .7096, .5475]),
    ("ukr_susp", [.7695, .8307, .7247, .7325, .7275, .9640, .7647, .6225]),
    ("twibot20", [.8688, .9473, .8600, .8431, .8025, .7115, .9487, .6895]),
    ("cp_hk",    [.6808, .7579, .7629, .6827, .6412, .6025, .7378, .9055]),
]
LADDER = [
    ("L1",    [.9480, .9730, .8740, .8490, .8280, .7710, .9210, .7240]),
    ("L2",    [.9450, .9800, .8850, .8430, .8280, .7750, .9250, .7260]),
    ("L3",    [.9410, .9780, .9150, .8300, .8150, .7770, .9270, .7200]),
    ("L4",    [.9344, .9753, .9093, .9113, .8297, .7768, .9234, .7235]),
    ("L5",    [.9346, .9754, .9086, .9102, .9259, .7693, .9254, .7261]),
    ("L6",    [.9325, .9744, .9073, .9106, .9241, .9340, .9242, .7239]),
    ("L7",    [.9321, .9748, .9033, .9076, .9198, .9256, .9377, .7267]),
    ("all8",  [.9340, .9750, .9080, .9060, .9200, .9310, .9370, .8670]),
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

    single_sorted = sorted(SINGLE, key=lambda m: float(np.mean(gaps(m[1]))))
    n_s = len(single_sorted)
    models = single_sorted + LADDER
    # single at 0..n_s-1, one blank slot, ladder at n_s+1..
    positions = list(range(n_s)) + [n_s + 1 + i for i in range(len(LADDER))]

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
                   s=42, edgecolor="0.25", linewidth=0.5, zorder=3)

    ax.axvline(n_s, color="0.6", lw=1.0, ls="--", zorder=1)   # group divider
    ax.set_xticks(positions)
    ax.set_xticklabels([m[0] for m in models], rotation=25, ha="right")
    ax.set_xlim(-0.7, positions[-1] + 0.7)
    ax.grid(axis="x", visible=False)

    top = float((best[None, :] - A).max())
    ax.set_ylabel("AUC gap to the best model on that graph  (0 = best)")
    ax.set_ylim(top * 1.05, -top * 0.03)                     # invert: best at top
    ax.set_title("How far each model trails the best on each test graph — specialists vs ladder\n"
                 "(best over all 16 models; single-source sorted by mean gap ↑; ladder L1→all8; 1 seed)",
                 fontsize=11, pad=26)

    tb = ax.get_xaxis_transform()  # x=data, y=axes fraction
    ax.text((n_s - 1) / 2, 1.005, "single-source specialists", transform=tb,
            ha="center", va="bottom", fontsize=10.5, color="0.3")
    ax.text(n_s + 1 + (len(LADDER) - 1) / 2, 1.005, "merged ladder", transform=tb,
            ha="center", va="bottom", fontsize=10.5, color="0.3")

    handles = [Line2D([0], [0], linestyle="none", marker="o", markerfacecolor=COLOR[g],
                      markeredgecolor="0.25", markersize=8, label=g) for g in GRAPHS]
    ax.legend(handles=handles, title="test graph", bbox_to_anchor=(1.005, 1.0),
              loc="upper left", frameon=False, fontsize=9, handletextpad=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"nm_ladder_plus_ss_delta_boxplot.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote nm_ladder_plus_ss_delta_boxplot.(pdf|png) to {OUT}")
    print("single order (mean-gap asc):", [m[0] for m in single_sorted])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
