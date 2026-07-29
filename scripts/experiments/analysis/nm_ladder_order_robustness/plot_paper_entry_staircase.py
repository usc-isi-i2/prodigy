#!/usr/bin/env python3
"""Paper figure: the entry-aligned staircase, sized for a 4-page LoG extended abstract.

A compact restyling of fig_entry_aligned() in plot_order_robustness.py. Same data and
same construction -- every (graph, order) AUC curve re-indexed by rungs relative to that
graph's own entry into the mixture, overlaid, with the mean and min/max band. Differences
from the analysis version, all for print:

  * 2.7:1 aspect at single-column width instead of 1.6:1, serif to match the body text
  * no title/subtitle (the LaTeX caption carries them)
  * the annotated jump is the PAIRED mean over the 21 measurable entry events, which is
    the number Table 3 reports (+.095). The analysis figure annotates
    mean(rel=0) - mean(rel=-1) over an unbalanced set of pairs (+.099); printing that
    next to Table 3 would read as an inconsistency.

Reads data/nm_ladder_order_robustness_long.csv. Writes the PDF straight into the paper
tree so there is no copy step to forget.
  /opt/homebrew/bin/python3.11 plot_paper_entry_staircase.py
"""
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "nm_ladder_order_robustness_long.csv")
OUT = os.path.abspath(os.path.join(
    HERE, "..", "..", "..", "..", "..", "paper", "LoG_extended_abstract",
    "GFM_LoG_EA", "figures"))

INK = "#0b0b0b"
MUTED = "#6f6d68"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
CORAL = "#d85a30"
GRAY = "#8f8d87"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
})


def load():
    rows = []
    with open(DATA, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["auc"] == "":
                continue
            rows.append(dict(order=r["order"], graph=r["test_graph"],
                             auc=float(r["auc"]), rel=int(r["rel_to_entry"])))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load()

    by_pair = defaultdict(dict)                       # (graph, order) -> {rel: auc}
    for r in rows:
        by_pair[(r["graph"], r["order"])][r["rel"]] = r["auc"]

    at_rel = defaultdict(list)
    for series in by_pair.values():
        for rel, auc in series.items():
            at_rel[rel].append(auc)
    keep = [rel for rel in range(-4, 8) if len(at_rel.get(rel, [])) >= 6]
    mean = [float(np.mean(at_rel[rel])) for rel in keep]
    lo = [float(np.min(at_rel[rel])) for rel in keep]
    hi = [float(np.max(at_rel[rel])) for rel in keep]

    # paired entry deltas: the 21 (graph, order) pairs that have a rung before entry
    deltas = [s[0] - s[-1] for s in by_pair.values() if 0 in s and -1 in s]
    n_pos = sum(d > 0 for d in deltas)
    print(f"[entry events] {n_pos}/{len(deltas)} positive, paired mean "
          f"{np.mean(deltas):+.4f}, median {np.median(deltas):+.4f}")

    fig, ax = plt.subplots(figsize=(6.9, 1.85), dpi=200)

    for series in by_pair.values():
        xs = sorted(series)
        ax.plot(xs, [series[x] for x in xs], color=GRAY, lw=0.6, alpha=0.30, zorder=2)

    ax.fill_between(keep, lo, hi, color=BLUE, alpha=0.10, zorder=3, linewidth=0)
    ax.plot(keep, mean, color=INK, lw=2.0, zorder=6, marker="o", ms=3.6,
            markerfacecolor=INK, markeredgecolor="white", markeredgewidth=0.8)

    ax.axvline(0, color=CORAL, lw=1.1, ls=(0, (3, 2)), zorder=4)
    ax.annotate("enters the mixture", xy=(0.16, 0.60), fontsize=7.6, color=CORAL,
                ha="left", va="center")

    m = dict(zip(keep, mean))
    ax.annotate(f"mean entry jump {np.mean(deltas):+.3f}",
                xy=(0, m[0]), xytext=(1.5, m[0] - 0.085), fontsize=7.8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.9,
                                shrinkA=0, shrinkB=3))

    ax.set_xlim(min(keep) - 0.35, max(keep) + 0.35)
    ax.set_xticks(keep)
    ax.set_ylim(0.53, 1.0)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("rungs relative to the target graph's own entry "
                  "($<0$ held out, $>0$ in mixture)", fontsize=8.2, color=INK)
    ax.set_ylabel("NM AUC", fontsize=8.2, color=INK)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=7.6)

    ax.legend(handles=[
        Line2D([0], [0], color=GRAY, lw=1.0, alpha=0.5,
               label="one (graph, order) pair"),
        Line2D([0], [0], color=INK, lw=2.0, marker="o", markerfacecolor=INK,
               markeredgecolor="white", ms=4, label="mean of 24 pairs"),
    ], loc="lower right", frameon=False, fontsize=7.4, handlelength=2.0,
        borderpad=0.2, labelspacing=0.3)

    fig.tight_layout(pad=0.25)
    out = os.path.join(OUT, "entry_staircase.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
