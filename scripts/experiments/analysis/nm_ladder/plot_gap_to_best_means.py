#!/usr/bin/env python3
"""NM interpolation-ladder — two-line summary, in GAP-TO-BEST (regret) terms.

Same layout as plot_nm_ladder_means.py, but each graph's value is its gap to the
best model on that graph instead of its absolute AUC. "Best" = the highest NM AUC
any of the 16 models reaches on that graph (the 8 ladder rungs + the 8 single-source
specialists) — i.e. the per-task frontier, which is ~= the specialist diagonal.

Two lines as the merge grows one source per rung:
  - "in training"  : mean gap-to-best over graphs already IN the merge (in-dist mean).
  - "all 8 graphs" : mean gap-to-best over the fixed 8-graph eval set (incl. held-out).
The all-8 line rises toward the in-dist line as coverage grows; the shaded gap between
them is the out-of-distribution penalty, closing to 0 at rung 8. Both converge NOT at
0 but at the residual in-domain regret (~ -0.020) — the "generalist tax" the merged
model still pays vs a per-graph specialist.

Values mirror nm_ladder_full.csv (ladder) + nm_single_source_matrix.csv (specialists),
embedded here since those CSVs are gitignored. Writes nm_ladder_gap_to_best_means.pdf/png.
"""
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

CANON = ["ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
         "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter"]
ENTRY = {k: i + 1 for i, k in enumerate(CANON)}   # graph -> rung it enters the merge
RUNGS = [1, 2, 3, 4, 5, 6, 7, 8]
XTICKS = ["ukr", "+covid", "+midterm", "+cov-pol", "+elec '20",
          "+ukr-susp", "+twibot", "+cp-hk\n(all 8)"]

# ladder: rung -> AUC on the 8 CANON graphs (matched-40k, within-balanced).
LADDER = {
    1: [0.9480, 0.9730, 0.8740, 0.8490, 0.8280, 0.7710, 0.9210, 0.7240],
    2: [0.9450, 0.9800, 0.8850, 0.8430, 0.8280, 0.7750, 0.9250, 0.7260],
    3: [0.9410, 0.9780, 0.9150, 0.8300, 0.8150, 0.7770, 0.9270, 0.7200],
    4: [0.9344, 0.9753, 0.9093, 0.9113, 0.8297, 0.7768, 0.9234, 0.7235],
    5: [0.9346, 0.9754, 0.9086, 0.9102, 0.9259, 0.7693, 0.9254, 0.7261],
    6: [0.9325, 0.9744, 0.9073, 0.9106, 0.9241, 0.9340, 0.9242, 0.7239],
    7: [0.9321, 0.9748, 0.9033, 0.9076, 0.9198, 0.9256, 0.9377, 0.7267],
    8: [0.9340, 0.9750, 0.9080, 0.9060, 0.9200, 0.9310, 0.9370, 0.8670],
}
# single-source specialists: train graph -> AUC on the 8 CANON graphs.
SPEC = [
    [0.9470, 0.9730, 0.8811, 0.8394, 0.8262, 0.7894, 0.9218, 0.7140],
    [0.9264, 0.9805, 0.8842, 0.8498, 0.8345, 0.7861, 0.9258, 0.7196],
    [0.7970, 0.8790, 0.9252, 0.8353, 0.8047, 0.6440, 0.8613, 0.6256],
    [0.6301, 0.6985, 0.6878, 0.9145, 0.7833, 0.5510, 0.7374, 0.5444],
    [0.6018, 0.6551, 0.6804, 0.7872, 0.9519, 0.5627, 0.7096, 0.5475],
    [0.7695, 0.8307, 0.7247, 0.7325, 0.7275, 0.9640, 0.7647, 0.6225],
    [0.8688, 0.9473, 0.8600, 0.8431, 0.8025, 0.7115, 0.9487, 0.6895],
    [0.6808, 0.7579, 0.7629, 0.6827, 0.6412, 0.6025, 0.7378, 0.9055],
]


def main():
    # per-graph best over ALL 16 models (ladder rungs + specialists) = the frontier.
    best = [max(max(LADDER[rg][j] for rg in RUNGS), max(row[j] for row in SPEC))
            for j in range(len(CANON))]

    # gap-to-best per rung per graph (<= 0)
    gap = {rg: [LADDER[rg][j] - best[j] for j in range(len(CANON))] for rg in RUNGS}
    x = list(range(len(RUNGS)))
    overall, included = [], []
    for ri, rg in enumerate(RUNGS):
        overall.append(sum(gap[rg]) / len(CANON))
        inc = [j for j, k in enumerate(CANON) if ENTRY[k] <= rg]  # already in the merge
        included.append(sum(gap[rg][j] for j in inc) / len(inc))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=200)

    ax.axhline(0.0, color="#c3c2b7", lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.annotate("best on each graph  (regret = 0)", xy=(3.4, 0.003),
                ha="center", va="bottom", fontsize=8.4, color=MUTED)

    # shaded gap = out-of-distribution penalty
    ax.fill_between(x, included, overall, color=BLUE, alpha=0.10, zorder=1, linewidth=0)

    ax.plot(x, included, color=BLUE, lw=2.6, zorder=5, marker="o", ms=6.5,
            markerfacecolor=BLUE, markeredgecolor="white", markeredgewidth=1.2)
    ax.plot(x, overall, color=INK, lw=2.6, zorder=6, marker="s", ms=6,
            markerfacecolor=INK, markeredgecolor="white", markeredgewidth=1.2)

    # direct labels at the right edge (lines converge at rung 8, so offset vertically)
    ax.annotate("in training\n(in-dist. mean)", xy=(x[-1] + 0.12, included[-1] + 0.006),
                ha="left", va="center", fontsize=9.5, color=BLUE, fontweight="bold")
    ax.annotate("all 8 graphs\n(incl. held-out)", xy=(x[-1] + 0.12, overall[-1] - 0.012),
                ha="left", va="center", fontsize=9.5, color=INK, fontweight="bold")

    # endpoint value labels
    ax.annotate(f"{included[0]:.3f}", xy=(x[0], included[0]), xytext=(12, 7),
                textcoords="offset points", ha="left", fontsize=8.6, color=BLUE,
                fontweight="bold")
    ax.annotate(f"{overall[0]:+.3f}", xy=(x[0], overall[0]), xytext=(0, -14),
                textcoords="offset points", ha="center", fontsize=8.6, color=INK,
                fontweight="bold")
    ax.annotate(f"{overall[-1]:+.3f}", xy=(x[-1], overall[-1]), xytext=(4, -14),
                textcoords="offset points", ha="center", fontsize=8.6, color=INK,
                fontweight="bold")

    # OOD-penalty arrow (widest at the left)
    gi = 1
    ax.annotate("", xy=(gi, included[gi]), xytext=(gi, overall[gi]),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.1))
    ax.annotate("out-of-distribution\npenalty (held-out graphs)",
                xy=(gi + 0.12, (included[gi] + overall[gi]) / 2),
                ha="left", va="center", fontsize=8.8, color=MUTED)
    ax.annotate("both lines converge to the\nresidual in-domain regret (-0.020)",
                xy=(4.75, -0.075), ha="left", va="center", fontsize=8.4, color=MUTED)

    ax.set_xlim(-0.5, 8.55)
    ax.set_ylim(-0.092, 0.012)
    ax.set_xticks(x)
    ax.set_xticklabels(XTICKS, fontsize=9.2)
    ax.set_xlabel("SSL pre-training graph  (one source added per rung, merge grows to the right)",
                  fontsize=10.5, color=INK)
    ax.set_ylabel("mean gap to best NM AUC   (0 = per-graph best)", fontsize=10.5, color=INK)
    ax.set_yticks([-0.08, -0.06, -0.04, -0.02, 0.00])
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title("Adding sources closes the gap to the best per-task model",
                 fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=26)
    ax.text(0.0, 1.02, "NM  3-shot / 30-way  ·  matched step 40k  ·  gap = AUC − best "
            "per graph (best over all 16: ladder rungs + single-source specialists)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.6, color=MUTED)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(HERE, "figures", f"nm_ladder_gap_to_best_means.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)

    print("\nrung | in-dist gap | all-8 gap | OOD penalty")
    for ri, rg in enumerate(RUNGS):
        print(f"  {rg}  |   {included[ri]:+.4f}  |  {overall[ri]:+.4f} | "
              f"{included[ri] - overall[ri]:.4f}")


if __name__ == "__main__":
    main()
