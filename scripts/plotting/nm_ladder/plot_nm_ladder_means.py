#!/usr/bin/env python3
"""NM interpolation-ladder — two-line summary figure.

Collapses the 8-graph ladder to two means as the SSL pre-training merge grows one
source per rung:
  - "all 8 graphs"    : mean NM AUC over the fixed 8-graph eval set (incl. held-out).
  - "in training"     : mean over only the graphs already IN the merge at that rung
                        (entry rung <= current rung) -- the in-distribution mean.
The in-distribution mean stays high and ~flat; the all-8 mean rises to meet it as
coverage grows. They coincide at rung 8 (everything in-distribution). The shaded gap
between them is the out-of-distribution penalty, shrinking as sources are added.

Shares data with plot_nm_ladder.py (nm_ladder_full.csv within_balanced rows; embedded
fallback since the CSV is gitignored). Writes nm_ladder_means.pdf/.png here.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

BLUE = "#2a78d6"   # in-distribution (in training)
INK = "#0b0b0b"    # all-8 mean
MUTED = "#898781"
GRID = "#e1e0d9"

CANON = [
    "ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
    "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter",
]
# graph -> rung at which it ENTERS the merge (one per rung).
ENTRY = {k: i + 1 for i, k in enumerate(CANON)}
RUNGS = [1, 2, 3, 4, 5, 6, 7, 8]
XTICKS = ["ukr", "+covid", "+midterm", "+cov-pol", "+elec '20",
          "+ukr-susp", "+twibot", "+cp-hk\n(all 8)"]

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
    os.path.join(HERE, "..", "..", "experiments", "nm_ladder_fillin", "nm_ladder_full.csv"),
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
            if all(rg in table for rg in RUNGS):
                src = os.path.relpath(path, HERE)
                break
            table = None
    if table is None:
        table = {rg: dict(zip(CANON, vals)) for rg, vals in EMBED.items()}
    print(f"[data] {src}")
    return {k: [table[rg][k] for rg in RUNGS] for k in CANON}


def main():
    series = load()
    x = list(range(len(RUNGS)))

    overall, included = [], []
    for ri, rg in enumerate(RUNGS):
        overall.append(sum(series[k][ri] for k in CANON) / len(CANON))
        inc = [k for k in CANON if ENTRY[k] <= rg]        # graphs already in the merge
        included.append(sum(series[k][ri] for k in inc) / len(inc))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=200)

    # shaded gap = out-of-distribution penalty
    ax.fill_between(x, included, overall, color=BLUE, alpha=0.10, zorder=1,
                    linewidth=0)

    # in-distribution mean (graphs already added)
    ax.plot(x, included, color=BLUE, lw=2.6, zorder=5, marker="o", ms=6.5,
            markerfacecolor=BLUE, markeredgecolor="white", markeredgewidth=1.2)
    # all-8 mean (incl. held-out)
    ax.plot(x, overall, color=INK, lw=2.6, zorder=6, marker="s", ms=6,
            markerfacecolor=INK, markeredgecolor="white", markeredgewidth=1.2)

    # direct labels at the right edge
    ax.annotate("in training\n(in-dist. mean)", xy=(x[-1] + 0.12, included[-1] + 0.006),
                ha="left", va="center", fontsize=9.5, color=BLUE, fontweight="bold")
    ax.annotate("all 8 graphs\n(incl. held-out)", xy=(x[-1] + 0.12, overall[-1] - 0.010),
                ha="left", va="center", fontsize=9.5, color=INK, fontweight="bold")

    # endpoint value labels
    ax.annotate(f"{included[0]:.3f}", xy=(x[0], included[0]), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=8.6,
                color=BLUE, fontweight="bold")
    ax.annotate(f"{overall[0]:.3f}", xy=(x[0], overall[0]), xytext=(0, -14),
                textcoords="offset points", ha="center", fontsize=8.6,
                color=INK, fontweight="bold")
    ax.annotate(f"{overall[-1]:.3f}", xy=(x[-1], overall[-1]), xytext=(0, -14),
                textcoords="offset points", ha="center", fontsize=8.6,
                color=INK, fontweight="bold")

    # gap annotation (widest at the left)
    gi = 1
    ax.annotate("", xy=(gi, included[gi]), xytext=(gi, overall[gi]),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.1))
    ax.annotate("out-of-distribution\npenalty (held-out graphs)",
                xy=(gi + 0.12, (included[gi] + overall[gi]) / 2),
                ha="left", va="center", fontsize=8.8, color=MUTED)
    ax.annotate("converge\n(all in-dist.)", xy=(x[-1] - 0.30, overall[-1] - 0.028),
                ha="right", va="top", fontsize=8.6, color=MUTED)

    ax.set_xlim(-0.5, 8.55)
    ax.set_ylim(0.84, 0.975)
    ax.set_xticks(x)
    ax.set_xticklabels(XTICKS, fontsize=9.2)
    ax.set_xlabel("SSL pre-training graph  (one source added per rung, merge grows to the right)",
                  fontsize=10.5, color=INK)
    ax.set_ylabel("mean NM AUC  (3-shot, 30-way)", fontsize=10.5, color=INK)
    ax.set_yticks([0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96])
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title("Adding sources closes the in-/out-of-distribution gap",
                 fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=26)
    ax.text(0.0, 1.02, "NM  3-shot / 30-way  ·  matched step 40k  ·  within-balanced "
            "sampling  ·  in-dist. mean = graphs already in the merge",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=MUTED)

    legend_handles = [
        Line2D([0], [0], color=BLUE, lw=2.6, marker="o", markerfacecolor=BLUE,
               markeredgecolor="white", ms=7, label="in training (in-dist. mean)"),
        Line2D([0], [0], color=INK, lw=2.6, marker="s", markerfacecolor=INK,
               markeredgecolor="white", ms=7, label="all 8 graphs (incl. held-out)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False,
              fontsize=9, handlelength=2.4, borderaxespad=0.6)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(HERE, f"nm_ladder_means.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)

    print("\nrung | in-dist mean | all-8 mean | gap")
    for ri, rg in enumerate(RUNGS):
        print(f"  {rg}  |    {included[ri]:.3f}    |   {overall[ri]:.3f}  | "
              f"{included[ri] - overall[ri]:.3f}")


if __name__ == "__main__":
    main()
