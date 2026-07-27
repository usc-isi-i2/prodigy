#!/usr/bin/env python3
"""Downstream ladder — aggregate performance vs. number of pre-training sources.

The analogue of ``analysis/nm_ladder/plot_nm_ladder_means.py``, on the downstream tasks.
Collapses each rung to two means:

  * "all eval graphs"  -- mean over the full eval set for that task, held-out included;
  * "graphs in the merge" -- mean over only the graphs already IN the merge at that rung
                             (the in-distribution mean).

Bands are the min/max across the three graph orders, NOT a confidence interval: with
n = 3 orders a CI would be meaningless, and the order-robustness analysis made the same
call. The band is therefore the observed spread, and a wide band means the orders
disagree at that rung.

What to look for: on the NM objective these two lines converge upward as coverage grows
(the OOD penalty closing). Downstream the story is different, and the figure is meant to
let that difference be read directly rather than asserted -- in particular, the
in-distribution mean drifting DOWN as sources are added is budget dilution, and it can
coexist with a positive entry effect (see plot_downstream_event_study.py).

Writes figures/nm_ladder_downstream_means.pdf/.png.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figures")

BLUE = "#2a78d6"   # in-distribution (in the merge)
INK = "#0b0b0b"    # all eval graphs
MUTED = "#898781"
GRID = "#e1e0d9"

PANELS = [
    ("slp", "Static link prediction", "AUC (degree-matched)"),
    ("pl", "Node classification", "ROC-AUC (10-shot)"),
    ("reg", "Node regression", "Spearman (10-shot)"),
]
RUNGS = list(range(1, 9))


def band(ax, xs, per_order, color, label, ls="-"):
    """Mean line across orders with a min/max band (n=3: spread, not a CI)."""
    mean, lo, hi = [], [], []
    for x in xs:
        vals = [per_order[o][x] for o in per_order if x in per_order[o]]
        mean.append(sum(vals) / len(vals))
        lo.append(min(vals))
        hi.append(max(vals))
    ax.fill_between(xs, lo, hi, color=color, alpha=0.13, lw=0, zorder=2)
    ax.plot(xs, mean, color=color, lw=2.2, ls=ls, zorder=4, label=label,
            solid_capstyle="round")


def main():
    df = pd.read_csv(os.path.join(DATA, "nm_ladder_downstream_long.csv"))
    df = df[df["primary"] == 1]

    os.makedirs(FIGS, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    for ax, (task, title, ylab) in zip(axes, PANELS):
        sub = df[df["task"] == task]
        all_mean, in_mean = {}, {}
        for order, g in sub.groupby("order"):
            all_mean[order] = g.groupby("rung")["value"].mean().to_dict()
            gi = g[g["in_merge"] == 1]
            in_mean[order] = gi.groupby("rung")["value"].mean().to_dict()

        band(ax, RUNGS, all_mean, INK, "all eval graphs")
        # the in-merge mean is undefined at rungs where nothing has entered yet for a
        # given task (e.g. static LP has no eligible graph in the merge at order-C r1)
        xs_in = [r for r in RUNGS if all(r in in_mean[o] for o in in_mean)]
        band(ax, xs_in, in_mean, BLUE, "graphs in the merge")

        ax.set_title(title, fontsize=11, color=INK, pad=8)
        ax.set_xlabel("number of sources in pre-training", fontsize=9, color=MUTED)
        ax.set_ylabel(ylab, fontsize=9, color=MUTED)
        ax.set_xticks(RUNGS)
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)

    handles = [
        Line2D([], [], color=INK, lw=2.2, label="all eval graphs (held-out included)"),
        Line2D([], [], color=BLUE, lw=2.2, label="only graphs in the merge"),
        Line2D([], [], color=MUTED, lw=6, alpha=0.25,
               label="min–max across the 3 graph orders"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Downstream performance vs. pre-training corpus size "
                 "(mean over eval graphs, matched 40k)", fontsize=12.5, color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))

    for ext in ("pdf", "png"):
        out = os.path.join(FIGS, f"nm_ladder_downstream_means.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
