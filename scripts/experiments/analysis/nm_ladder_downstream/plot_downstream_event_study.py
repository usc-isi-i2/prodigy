#!/usr/bin/env python3
"""Downstream ladder — entry-aligned event study across all three graph orders.

Each (task, eval graph, order) gives one series. The x axis is ``rel_to_entry`` = rung
minus the rung at which that graph joined the training merge, so series from different
orders are directly comparable: x < 0 is held out, x >= 0 is in the merge, and x = 0 is
the entry event itself.

Series are normalised to their own value at x = -1 (the rung just before entry), so
every line starts at 0 and the vertical move between -1 and 0 IS the entry effect.
Only series that actually have a "before" observation can be drawn -- a graph entering
at rung 1 is a single-source specialist with nothing to compare against, which is also
why the sign test has 13/11/30 events rather than 24 per task.

The bold line is the mean over series; the annotation is a one-sided sign test on the
entry deltas, computed here rather than hardcoded.

Companion to ``plot_downstream_trajectory.py`` (which shows the raw levels for order A).
Writes figures/nm_ladder_downstream_event_study.pdf/.png.
"""
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figures")

BLUE = "#2a78d6"
GRAY = "#8f8d87"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

PANELS = [
    ("slp", "Static link prediction", "AUC"),
    ("pl", "Node classification", "ROC-AUC"),
    ("reg", "Node regression", "Spearman"),
]


def sign_test_ge(k, n):
    """One-sided P(X >= k) for X ~ Binomial(n, 0.5)."""
    if n == 0:
        return float("nan")
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def main():
    df = pd.read_csv(os.path.join(DATA, "nm_ladder_downstream_long.csv"))
    df = df[df["primary"] == 1]

    os.makedirs(FIGS, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharex=True)

    for ax, (task, title, metric) in zip(axes, PANELS):
        sub = df[df["task"] == task]
        series, deltas = [], []

        for _, g in sub.groupby(["dataset", "order", "target"], dropna=False):
            g = g.sort_values("rel_to_entry")
            base = g[g["rel_to_entry"] == -1]
            if base.empty or (g["rel_to_entry"] == 0).sum() == 0:
                continue                      # no "before": rung-1 specialist entry
            b = float(base["value"].iloc[0])
            xs = g["rel_to_entry"].tolist()
            ys = [v - b for v in g["value"]]
            series.append((xs, ys))
            deltas.append(float(g.loc[g["rel_to_entry"] == 0, "value"].iloc[0]) - b)

        for xs, ys in series:
            ax.plot(xs, ys, color=GRAY, lw=0.9, alpha=0.5, zorder=2)

        # mean across series at each offset
        acc = {}
        for xs, ys in series:
            for x, y in zip(xs, ys):
                acc.setdefault(x, []).append(y)
        mx = sorted(acc)
        ax.plot(mx, [sum(acc[x]) / len(acc[x]) for x in mx], color=BLUE, lw=2.4,
                zorder=4, solid_capstyle="round")

        ax.axvline(0, color=INK, lw=0.9, ls="--", alpha=0.55, zorder=3)
        ax.axhline(0, color=GRID, lw=1.0, zorder=1)

        pos, n = sum(1 for d in deltas if d > 0), len(deltas)
        p = sign_test_ge(pos, n)
        mean_d = sum(deltas) / n if n else float("nan")
        verdict = "significant" if p < 0.05 else "n.s."
        ax.set_title(title, fontsize=11, color=INK, pad=8)
        ax.annotate(f"entry $\\Delta$ > 0 in {pos}/{n} events\n"
                    f"sign test p = {p:.3f} ({verdict})\nmean $\\Delta$ = {mean_d:+.3f}",
                    xy=(0.03, 0.03), xycoords="axes fraction", fontsize=8.5,
                    color=INK if p < 0.05 else MUTED, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.4", fc="white",
                              ec=BLUE if p < 0.05 else GRID, lw=1.0, alpha=0.9))

        ax.set_xlabel("rungs relative to the graph's entry into the merge",
                      fontsize=9, color=MUTED)
        ax.set_ylabel(f"$\\Delta$ {metric} vs. the rung before entry",
                      fontsize=9, color=MUTED)
        ax.set_xticks(range(-7, 8, 1))
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)

    handles = [
        Line2D([], [], color=GRAY, lw=0.9, alpha=0.6,
               label="one (graph × order) series"),
        Line2D([], [], color=BLUE, lw=2.4, label="mean over series"),
        Line2D([], [], color=INK, lw=0.9, ls="--", alpha=0.6,
               label="entry into the training merge"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Does adding a graph to pre-training help that graph downstream? "
                 "(entry-aligned, 3 graph orders)", fontsize=12.5, color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))

    for ext in ("pdf", "png"):
        out = os.path.join(FIGS, f"nm_ladder_downstream_event_study.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
