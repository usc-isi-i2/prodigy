#!/usr/bin/env python3
"""NM interpolation-ladder — per-source impact as three delta LINES (single axis).

Same role-decomposed deltas as plot_nm_ladder_deltas.py, drawn as three lines over
the sequence of added sources (rungs 2..8) instead of grouped bars:
  - newcomer   : the graph just added -- its own OOD->ID jump (BENEFIT).
  - incumbents : graphs already in training -- mean Delta (COST / dilution).
  - held-out   : graphs not yet added   -- mean Delta (interference).

One shared y-axis (no dual axis): the newcomer line swings up to +.165 while the
incumbent and held-out lines stay pinned within a +/-.006 band around zero -- that
scale gap IS the finding (big in-domain benefit, negligible effect on everything
else). A shaded band marks the +/-.006 envelope the two small lines never leave.

Shares data with the other nm_ladder plots (nm_ladder_full.csv within_balanced rows;
embedded fallback). Writes nm_ladder_deltas_lines.pdf/.png here.
"""
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

BLUE = "#2a78d6"    # newcomer benefit
CORAL = "#d85a30"   # incumbents (cost)
GRAY = "#8f8d87"    # held-out (interference)
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

CANON = [
    "ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
    "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter",
]
ENTRY = {k: i + 1 for i, k in enumerate(CANON)}
ADDED = ["+covid", "+midterm", "+cov-pol", "+elec '20", "+ukr-susp", "+twibot", "+cp-hk"]

EMBED = {
    1: [0.9480, 0.9730, 0.8740, 0.8490, 0.8280, 0.7710, 0.9210, 0.7240],
    2: [0.9450, 0.9800, 0.8850, 0.8430, 0.8280, 0.7750, 0.9250, 0.7260],
    3: [0.9410, 0.9780, 0.9150, 0.8300, 0.8150, 0.7770, 0.9270, 0.7200],
    4: [0.9344, 0.9753, 0.9093, 0.9113, 0.8297, 0.7768, 0.9234, 0.7235],
    5: [0.9346, 0.9754, 0.9086, 0.9102, 0.9259, 0.7693, 0.9254, 0.7261],
    6: [0.9325, 0.9744, 0.9073, 0.9106, 0.9241, 0.9340, 0.9242, 0.7239],
    7: [0.9321, 0.9748, 0.9033, 0.9076, 0.9198, 0.9256, 0.9377, 0.7267],
    8: [0.9340, 0.9750, 0.9080, 0.9060, 0.9200, 0.9310, 0.9370, 0.8670],
}
CSV_CANDIDATES = [
    os.path.join(HERE, "nm_ladder_full.csv"),
    os.path.join(HERE, "data", "nm_ladder_full.csv"),
]


def load():
    table, src = None, "embedded fallback"
    for path in CSV_CANDIDATES:
        if os.path.exists(path):
            table = {}
            with open(path, newline="") as fh:
                for r in csv.DictReader(fh):
                    if r.get("sampling", "within_balanced") != "within_balanced":
                        continue
                    try:
                        table[int(r["rung"])] = {k: float(r[k]) for k in CANON}
                    except (KeyError, ValueError):
                        continue
            if all(rg in table for rg in range(1, 9)):
                src = os.path.relpath(path, HERE)
                break
            table = None
    if table is None:
        table = {rg: dict(zip(CANON, vals)) for rg, vals in EMBED.items()}
    print(f"[data] {src}")
    return {k: [table[rg][k] for rg in range(1, 9)] for k in CANON}


def main():
    series = load()
    newc, inc, hel = [], [], []
    for r in range(2, 9):
        d = {c: series[c][r - 1] - series[c][r - 2] for c in CANON}
        newc.append(d[CANON[r - 1]])
        incs = [c for c in CANON if ENTRY[c] < r]
        hels = [c for c in CANON if ENTRY[c] > r]
        inc.append(sum(d[c] for c in incs) / len(incs))
        hel.append(sum(d[c] for c in hels) / len(hels) if hels else math.nan)

    x = list(range(len(ADDED)))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(9.4, 5.7), dpi=200)

    # +/-.006 envelope the two small lines stay within
    ax.axhspan(-0.006, 0.006, color=GRAY, alpha=0.10, zorder=0, linewidth=0)
    ax.axhline(0, color="#c3c2b7", lw=1.0, zorder=1)

    # held-out (drop the trailing NaN so the line just stops)
    xh = [xi for xi, v in zip(x, hel) if not math.isnan(v)]
    yh = [v for v in hel if not math.isnan(v)]
    ax.plot(xh, yh, color=GRAY, lw=1.8, ls=(0, (4, 2)), marker="^", ms=6,
            markerfacecolor="white", markeredgecolor=GRAY, markeredgewidth=1.4, zorder=4)
    # incumbents
    ax.plot(x, inc, color=CORAL, lw=2.0, marker="s", ms=6, markerfacecolor=CORAL,
            markeredgecolor="white", markeredgewidth=1.1, zorder=5)
    # newcomer benefit
    ax.plot(x, newc, color=BLUE, lw=2.8, marker="o", ms=7.5, markerfacecolor=BLUE,
            markeredgecolor="white", markeredgewidth=1.3, zorder=6)
    for xi, v in zip(x, newc):
        ax.annotate(f"+{v:.3f}", xy=(xi, v), xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=BLUE, fontweight="bold")

    ax.annotate("±0.006 band", xy=(6.45, -0.006), ha="right", va="top", fontsize=8, color=MUTED)

    ax.set_xlim(-0.35, 6.55)
    ax.set_ylim(-0.02, 0.185)
    ax.set_xticks(x)
    ax.set_xticklabels(ADDED, fontsize=9.6)
    ax.set_xlabel("source added to the SSL pre-training merge", fontsize=10.5, color=INK)
    ax.set_ylabel("Δ NM AUC  (this rung − rung below)", fontsize=10.5, color=INK)
    ax.set_yticks([0.00, 0.05, 0.10, 0.15])
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title("Per-source impact (Δ): the added graph gains big; the others barely move",
                 fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=26)
    ax.text(0.0, 1.02, "NM 3-shot / 30-way · matched step 40k · within-balanced · "
            "per-graph deltas averaged within each role",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=MUTED)

    handles = [
        Line2D([0], [0], color=BLUE, lw=2.8, marker="o", markerfacecolor=BLUE,
               markeredgecolor="white", ms=7.5, label="newcomer — benefit"),
        Line2D([0], [0], color=CORAL, lw=2.0, marker="s", markerfacecolor=CORAL,
               markeredgecolor="white", ms=6, label="incumbents — cost"),
        Line2D([0], [0], color=GRAY, lw=1.8, ls=(0, (4, 2)), marker="^",
               markerfacecolor="white", markeredgecolor=GRAY, ms=6,
               label="held-out — interference"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9,
              handlelength=2.2, borderaxespad=0.8)

    for ext in ("pdf", "png"):
        out = os.path.join(HERE, "figures", f"nm_ladder_deltas_lines.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)

    print("\nadd        newcomer   incumbents   held-out")
    for i, a in enumerate(ADDED):
        h = f"{hel[i]:+.4f}" if not math.isnan(hel[i]) else "   -"
        print(f"  {a:<10}{newc[i]:+.4f}    {inc[i]:+.4f}    {h}")


if __name__ == "__main__":
    main()
