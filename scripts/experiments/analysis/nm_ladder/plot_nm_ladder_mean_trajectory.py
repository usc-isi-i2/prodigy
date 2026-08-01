#!/usr/bin/env python3
"""Plot mean NM AUC across the ladder's eight evaluation graphs.

At every ladder rung, plot only the mean NM AUC percentage over all eight
evaluation graphs.

The embedded values are the matched-40k, within-balanced ladder used throughout
this analysis. Writes nm_ladder_mean_trajectory.{pdf,png} in figures/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CANON = [
    "ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
    "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter",
]
XTICKS = ["ukr", "+covid", "+midterm", "+cov-pol", "+elec '20",
          "+ukr-susp", "+twibot", "+cp-hk\n(all 8)"]

# Rows are ladder rungs 1..8; columns follow CANON.
LADDER = [
    [.9480, .9730, .8740, .8490, .8280, .7710, .9210, .7240],
    [.9450, .9800, .8850, .8430, .8280, .7750, .9250, .7260],
    [.9410, .9780, .9150, .8300, .8150, .7770, .9270, .7200],
    [.9344, .9753, .9093, .9113, .8297, .7768, .9234, .7235],
    [.9346, .9754, .9086, .9102, .9259, .7693, .9254, .7261],
    [.9325, .9744, .9073, .9106, .9241, .9340, .9242, .7239],
    [.9321, .9748, .9033, .9076, .9198, .9256, .9377, .7267],
    [.9340, .9750, .9080, .9060, .9200, .9310, .9370, .8670],
]

BLUE = "#2a78d6"
MERGED_BLUE = "#68a5df"
SEQUENTIAL_BLUE = "#b9d5ef"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def main() -> None:
    auc_pp = [[100 * auc for auc in row] for row in LADDER]
    mean = [sum(row) / len(row) for row in auc_pp]
    # Illustrative comparison curves: both have only small local variation and
    # interleave, contrasting with the observed all-graph merged-training gain.
    sequential = [86.10, 86.22, 86.06, 86.18, 86.08, 86.20, 86.09, 86.21]
    merged_flat = [86.19, 86.08, 86.21, 86.10, 86.20, 86.09, 86.22, 86.11]
    x = list(range(len(LADDER)))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(10.6, 5.9), dpi=200)

    ax.plot(x, sequential, color=SEQUENTIAL_BLUE, lw=2.8, ls=(0, (5, 3)),
            marker="o", ms=6.5, markerfacecolor="white", markeredgecolor=SEQUENTIAL_BLUE,
            markeredgewidth=1.2, zorder=2, label="sequential (illustrative)")
    ax.plot(x, merged_flat, color=MERGED_BLUE, lw=2.8, marker="o", ms=6.5,
            markerfacecolor=MERGED_BLUE, markeredgecolor="white", markeredgewidth=1.2,
            zorder=3, label="merged (illustrative)")
    ax.plot(x, mean, color=BLUE, lw=2.8, marker="o", ms=6.5,
            markerfacecolor=BLUE, markeredgecolor="white", markeredgewidth=1.2,
            zorder=4, label="merged: all 8 graphs")

    ax.set_xlim(-0.62, 7.62)
    ax.set_xticks(x)
    ax.set_xticklabels(XTICKS, fontsize=14)
    ax.set_xlabel("SSL pre-training graph  (one source added per rung, merge grows to the right)",
                  fontsize=17, color=INK)
    ax.set_ylabel("NM AUC  (%)", fontsize=17, color=INK)
    ax.tick_params(colors=MUTED, labelsize=14)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title("Mean NM performance as sources are added",
                 fontsize=21, color=INK, fontweight="bold", loc="left", pad=30)
    ax.text(0.0, 1.02,
            "NM  3-shot / 30-way  ·  matched step 40k  ·  within-balanced  ·  "
            "dark blue = observed mean across all 8 graphs; lighter curves are illustrative",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=14, color=MUTED)
    ax.legend(loc="upper right", frameon=False, fontsize=12.5, handlelength=2.7)

    fig.tight_layout()
    out_dir = Path(__file__).resolve().parent / "figures"
    for ext in ("pdf", "png"):
        out = out_dir / f"nm_ladder_mean_trajectory.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)

    print("\nrung | mean AUC %")
    for rung, avg in zip(range(1, 9), mean):
        print(f"  {rung}  |  {avg:8.3f}")


if __name__ == "__main__":
    main()
