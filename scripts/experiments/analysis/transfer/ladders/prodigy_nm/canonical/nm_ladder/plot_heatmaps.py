#!/usr/bin/env python3
"""Static (matplotlib) versions of this conversation's ladder heatmaps, written into
./figures/ for the graph-ladder figure collection. Also copies the per-step delta plot.

Figures written to ./figures/:
  nm_ladder_plus_single_source_heatmap.pdf  -- 16x8 NM AUC (8 ladder rungs + 8 specialists)
  nm_ladder_regret_heatmap.pdf              -- 16x8 gap-to-best (AUC pts) + mean Δ + mean rank
  nm_ladder_per_step_delta.pdf/.png         -- copied from the parent folder
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

COLS = ["ukr", "covid", "mid", "cov_pol", "elec20", "ukr_susp", "twibot", "cp_hk"]
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
SINGLE = [
    ("S · ukr",        [.9470, .9730, .8811, .8394, .8262, .7894, .9218, .7140]),
    ("S · covid",      [.9264, .9805, .8842, .8498, .8345, .7861, .9258, .7196]),
    ("S · midterm",    [.7970, .8790, .9252, .8353, .8047, .6440, .8613, .6256]),
    ("S · cov_pol",    [.6301, .6985, .6878, .9145, .7833, .5510, .7374, .5444]),
    ("S · elec20",     [.6018, .6551, .6804, .7872, .9519, .5627, .7096, .5475]),
    ("S · ukr_susp",   [.7695, .8307, .7247, .7325, .7275, .9640, .7647, .6225]),
    ("S · twibot20",   [.8688, .9473, .8600, .8431, .8025, .7115, .9487, .6895]),
    ("S · cp_hk",      [.6808, .7579, .7629, .6827, .6412, .6025, .7378, .9055]),
]
ROWS = LADDER + SINGLE
LABELS = [r[0] for r in ROWS]
M = np.array([r[1] for r in ROWS])   # 16 x 8
NL = len(LADDER)
CORAL, TEAL, TEALBG, RED = "#D85A30", "#0F6E56", "#E1F5EE", "#791F1F"


def diag_cells():
    # "own-graph" cell per row: ladder entry (i, i) + specialist in-domain (NL+k, k)
    return [(i, i) for i in range(8)] + [(NL + k, k) for k in range(8)]


def auc_heatmap():
    fig, ax = plt.subplots(figsize=(9.2, 8.6))
    ax.imshow(M, cmap="Blues", vmin=0.54, vmax=0.98, aspect="auto")
    for i in range(16):
        for j in range(8):
            v = M[i, j]
            ax.text(j, i, f"{v:.3f}".lstrip("0"), ha="center", va="center",
                    color="white" if v > 0.86 else "#0C447C", fontsize=8)
    for (i, j) in diag_cells():
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=CORAL, lw=2))
    ax.axhline(NL - 0.5, color="#444441", lw=1.4)
    ax.set_xticks(range(8)); ax.set_xticklabels(COLS, fontsize=9)
    ax.set_yticks(range(16)); ax.set_yticklabels(LABELS, fontsize=9)
    ax.tick_params(length=0)
    ax.set_title("NM AUC — ladder rungs (top) + single-source specialists (bottom)\n"
                 "coral = own-graph cell (enters merge / in-domain); matched-40k, 1 seed",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "nm_ladder_plus_single_source_heatmap.pdf")
    plt.close(fig)


def regret_heatmap():
    best = M.max(axis=0)
    delta = (M - best) * 100.0            # <= 0, AUC points below column-best
    mag = -delta
    meanD = delta.mean(axis=1)            # <= 0
    ranks = np.zeros_like(M)
    for j in range(8):
        col = M[:, j]
        ranks[:, j] = [np.sum(col > v) + 1 + (np.sum(col == v) - 1) / 2 for v in col]
    meanR = ranks.mean(axis=1)

    fig = plt.figure(figsize=(12.0, 8.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[8, 1, 1], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    axd = fig.add_subplot(gs[1], sharey=ax)
    axr = fig.add_subplot(gs[2], sharey=ax)

    ax.imshow(mag, cmap="Reds", vmin=0, vmax=30, aspect="auto")
    for i in range(16):
        for j in range(8):
            if mag[i, j] < 0.05:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=TEALBG,
                                       edgecolor=TEAL, lw=2))
                ax.text(j, i, "0", ha="center", va="center", color=TEAL, fontsize=8)
            else:
                ax.text(j, i, f"−{mag[i, j]:.1f}", ha="center", va="center",
                        color="white" if mag[i, j] > 15 else RED, fontsize=7.5)
    ax.set_xticks(range(8)); ax.set_xticklabels(COLS, fontsize=9)
    ax.set_yticks(range(16)); ax.set_yticklabels(LABELS, fontsize=9)
    ax.tick_params(length=0)

    axd.imshow((-meanD).reshape(-1, 1), cmap="Reds", vmin=0, vmax=26, aspect="auto")
    for i in range(16):
        axd.text(0, i, f"−{-meanD[i]:.1f}", ha="center", va="center", fontweight="bold",
                 color="white" if -meanD[i] > 13 else RED, fontsize=8)
    axd.set_xticks([0]); axd.set_xticklabels(["mean Δ"], fontsize=9)
    axd.tick_params(length=0); plt.setp(axd.get_yticklabels(), visible=False)

    axr.imshow(meanR.reshape(-1, 1), cmap="Reds", vmin=4, vmax=14, aspect="auto")
    for i in range(16):
        axr.text(0, i, f"{meanR[i]:.1f}", ha="center", va="center", fontweight="bold",
                 color="white" if meanR[i] > 9 else RED, fontsize=8)
    axr.set_xticks([0]); axr.set_xticklabels(["mean rank"], fontsize=9)
    axr.tick_params(length=0); plt.setp(axr.get_yticklabels(), visible=False)

    for a in (ax, axd, axr):
        a.axhline(NL - 0.5, color="#444441", lw=1.4)
    ax.set_title("Gap to best per test graph (AUC points below column-best), + per-model "
                 "mean Δ and mean rank\nteal = best on that graph; matched-40k, 1 seed",
                 fontsize=11, loc="left")
    fig.savefig(FIG / "nm_ladder_regret_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def copy_existing():
    for name in ("nm_ladder_per_step_delta.pdf", "nm_ladder_per_step_delta.png"):
        src = HERE / name
        if src.exists():
            shutil.copy(src, FIG / name)


if __name__ == "__main__":
    auc_heatmap()
    regret_heatmap()
    copy_existing()
    print(f"wrote into {FIG}:")
    for p in sorted(FIG.iterdir()):
        print("  ", p.name)
