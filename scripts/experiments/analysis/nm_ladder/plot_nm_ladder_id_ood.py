#!/usr/bin/env python3
"""In-distribution vs out-of-distribution mean NM AUC across the ladder.

Distills the 8-rung interpolation ladder (plot_nm_ladder.py) to TWO lines. At each
rung, split the 8 single-source eval graphs by whether their source is already in the
training merge, and average:
  - in-distribution  = mean AUC over graphs already added (entry rung <= current rung)
  - out-of-distribution = mean AUC over graphs not yet added (entry rung > current rung)
The persistent vertical gap is the generalization deficit on unseen domains; each graph
crosses from the lower line to the upper line at the rung it enters training.

Note the two means are over CHANGING sets (a graph leaves the OOD pool when it enters),
so the point count n is annotated at each marker; at the final rung all 8 are
in-distribution and the OOD line ends. Reuses data + entry rungs from plot_nm_ladder.py.
Writes nm_ladder_id_ood.pdf/.png next to this file.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from plot_nm_ladder import (  # noqa: E402  (sibling module, same dir)
    load, GRAPHS, RUNGS, XTICKS, BLUE, GRAY, INK, MUTED, GRID,
)


def main():
    series = load()  # {key: [auc per rung]}
    idx = list(range(len(RUNGS)))

    id_mean, ood_mean, id_n, ood_n = [], [], [], []
    for ri, rung in enumerate(RUNGS):
        id_vals = [series[k][ri] for k, _l, e in GRAPHS if e <= rung]
        ood_vals = [series[k][ri] for k, _l, e in GRAPHS if e > rung]
        id_mean.append(sum(id_vals) / len(id_vals) if id_vals else None)
        ood_mean.append(sum(ood_vals) / len(ood_vals) if ood_vals else None)
        id_n.append(len(id_vals))
        ood_n.append(len(ood_vals))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=200)

    # gap shading where both means are defined (rungs 1..7)
    xi = [i for i in idx if ood_mean[i] is not None]
    ax.fill_between(xi, [ood_mean[i] for i in xi], [id_mean[i] for i in xi],
                    color=BLUE, alpha=0.07, zorder=1)

    # in-distribution line (solid blue, filled markers)
    ax.plot(idx, id_mean, color=BLUE, lw=2.6, zorder=4, solid_capstyle="round")
    ax.scatter(idx, id_mean, s=52, facecolor=BLUE, edgecolor="white",
               linewidth=1.4, zorder=5)

    # out-of-distribution line (dashed gray, open markers), ends when OOD pool empties
    xo = [i for i in idx if ood_mean[i] is not None]
    yo = [ood_mean[i] for i in xo]
    ax.plot(xo, yo, color=GRAY, lw=2.2, ls=(0, (4, 2)), zorder=4, solid_capstyle="round")
    ax.scatter(xo, yo, s=46, facecolor="white", edgecolor=GRAY, linewidth=1.6, zorder=5)

    # n annotations (sets change size): ID count above, OOD count below
    for i in idx:
        ax.annotate(f"n={id_n[i]}", xy=(i, id_mean[i]), xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.6, color=MUTED)
        if ood_mean[i] is not None:
            ax.annotate(f"n={ood_n[i]}", xy=(i, ood_mean[i]), xytext=(0, -9),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=7.6, color=MUTED)

    # gap callout at a mid rung
    gi = 3  # rung 4
    gap = id_mean[gi] - ood_mean[gi]
    ax.annotate("", xy=(gi, id_mean[gi]), xytext=(gi, ood_mean[gi]),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1), zorder=6)
    ax.annotate(f"gap {gap:.2f}", xy=(gi + 0.08, (id_mean[gi] + ood_mean[gi]) / 2),
                ha="left", va="center", fontsize=9, color=INK, fontweight="bold")

    # note why the OOD line ends before the last rung (identity is via the legend)
    ax.annotate("rung 8:\nnothing held out", xy=(idx[-1] - 0.02, 0.79),
                ha="center", va="center", fontsize=8.5, color=MUTED, style="italic")

    # axes / chrome
    ax.set_xlim(-0.5, 8.35)
    ax.set_ylim(0.70, 1.0)
    ax.set_xticks(idx)
    ax.set_xticklabels(XTICKS, fontsize=9.2)
    ax.set_xlabel("SSL pre-training graph  (one source added per rung, merge grows to the right)",
                  fontsize=10.5, color=INK)
    ax.set_ylabel("mean NM AUC  (3-shot, 30-way)", fontsize=10.5, color=INK)
    ax.set_yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title("In- vs out-of-distribution mean AUC: a persistent generalization gap",
                 fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=26)
    ax.text(0.0, 1.02, "NM  3-shot / 30-way  ·  matched step 40k  ·  within-balanced "
            "sampling  ·  n = graphs averaged (each pool changes as graphs enter)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=MUTED)

    legend_handles = [
        Line2D([0], [0], color=BLUE, lw=2.6, marker="o", markerfacecolor=BLUE,
               markeredgecolor="white", ms=7, label="in-distribution (trained-on)"),
        Line2D([0], [0], color=GRAY, lw=2.2, ls=(0, (4, 2)), marker="o",
               markerfacecolor="white", markeredgecolor=GRAY, ms=7,
               label="out-of-distribution (held out)"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=False,
              fontsize=9, handlelength=2.4, borderaxespad=0.6)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(HERE, "figures", f"nm_ladder_id_ood.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)

    print("\nrung   in-dist (n)      out-of-dist (n)     gap")
    for i, rung in enumerate(RUNGS):
        om = f"{ood_mean[i]:.3f} ({ood_n[i]})" if ood_mean[i] is not None else "  --   (0)"
        gap = f"{id_mean[i] - ood_mean[i]:+.3f}" if ood_mean[i] is not None else "  -- "
        print(f"  {rung}    {id_mean[i]:.3f} ({id_n[i]})      {om:16s}  {gap}")


if __name__ == "__main__":
    main()
