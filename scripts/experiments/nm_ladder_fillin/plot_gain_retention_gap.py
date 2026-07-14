#!/usr/bin/env python3
"""Gain-vs-retention figure, in GAP-TO-BEST (regret) terms.

Same design as plot_gain_retention.py, but every value is a graph's gap to the best
model on that graph (AUC - best-per-graph, best over all 16 models = the 8 ladder
rungs + the 8 single-source specialists) instead of its absolute AUC. So 0 = the best
achievable per graph (the frontier ~= the specialist diagonal), and everything sits at
or below it.

  (1) newcomers CLOSE the gap: each newly-added graph's coral arrow rises from its deep
      out-of-dist gap (at the previous rung) up to a near-0 in-training gap;
  (2) incumbents STAY near best: the mean gap over graphs already in training hugs 0
      (only ~-0.02), with a tight dispersion band.

Because best-per-graph is a per-column constant, each arrow's LENGTH equals the AUC
jump; the reframe just aligns every "after" point near 0, so you see each graph reach
~its own best on entering (and the residual gap it still pays vs its specialist).

Three band variants for the in-training dispersion:
  flat -> nm_ladder_gain_retention_gap.pdf ; std -> ..._gap_std.pdf ; minmax -> ..._gap_minmax.pdf
Values embedded (mirror nm_ladder_full.csv + nm_single_source_matrix.csv). matplotlib.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SHORT = ["ukr", "covid", "midterm", "cov_pol", "elec20", "ukr_susp", "twibot20", "cp_hk"]
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
BLUE, CORAL, DCORAL = "#185FA5", "#D85A30", "#993C1D"
BEST = [max(max(LADDER[r][j] for r in range(1, 9)), max(row[j] for row in SPEC))
        for j in range(8)]
# gap-to-best per rung per graph (<= 0)
GAP = {r: [LADDER[r][j] - BEST[j] for j in range(8)] for r in range(1, 9)}


def _stats(xs):
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
    return m, sd, min(xs), max(xs)


def make_figure(band: str, out: Path) -> None:
    rungs = list(range(1, 9))
    stat = {r: _stats(GAP[r][:r]) for r in rungs}          # over in-training graphs
    inc = {r: stat[r][0] for r in rungs}
    # newcomer at rung r: graph col (r-1). before = rung r-1 gap (still OOD), at x=r-1;
    # after = rung r gap (now in-training), at x=r.
    newcomer = {r: (GAP[r - 1][r - 1], GAP[r][r - 1]) for r in range(2, 9)}

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.axhline(0.0, color="#c3c2b7", lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.annotate("best on each graph  (gap = 0)", xy=(4.5, 0.004), ha="center",
                va="bottom", fontsize=8.6, color="#7a7873")

    if band == "flat":
        ax.axhspan(min(inc.values()), max(inc.values()), color=BLUE, alpha=0.08, zorder=0)
        band_label = "in-training mean range"
    elif band == "std":
        ax.fill_between(rungs, [inc[r] - stat[r][1] for r in rungs],
                        [inc[r] + stat[r][1] for r in rungs], color=BLUE, alpha=0.14,
                        zorder=0, lw=0)
        band_label = "in-training graphs: mean ± 1 std"
    elif band == "minmax":
        ax.fill_between(rungs, [stat[r][2] for r in rungs], [stat[r][3] for r in rungs],
                        color=BLUE, alpha=0.12, zorder=0, lw=0)
        band_label = "in-training graphs: min–max"

    # newcomer jumps: before (deep, OOD) at x=r-1 -> after (near best) at x=r
    for r, (b, a) in newcomer.items():
        ax.annotate("", xy=(r, a), xytext=(r - 1, b),
                    arrowprops=dict(arrowstyle="-|>", color=CORAL, lw=2.2,
                                    shrinkA=0, shrinkB=0), zorder=4)
        ax.plot(r - 1, b, "o", mfc="white", mec=CORAL, mew=1.6, ms=7, zorder=5)
        ax.plot(r, a, "o", color=CORAL, ms=7, zorder=5)
        # label the gap CLOSED near the (spread-out) before point, below-left
        ax.annotate(f"+{a - b:.2f}", xy=(r - 1, b), xytext=(-9, -1),
                    textcoords="offset points", fontsize=8.2, color=DCORAL,
                    va="center", ha="right")

    ax.plot(rungs, [inc[r] for r in rungs], "-o", color=BLUE, lw=2.4, ms=6, zorder=3)

    ax.set_xlabel("training-merge size (# source graphs)")
    ax.set_ylabel("gap to best NM AUC   (0 = per-graph best)")
    ax.set_xticks(rungs)
    ax.set_xticklabels(["1\nukr"] + [f"{r}\n+{SHORT[r - 1]}" for r in range(2, 9)])
    ax.set_xlim(0.4, 9.1)
    ax.set_ylim(-0.205, 0.02)
    ax.set_yticks([0.00, -0.04, -0.08, -0.12, -0.16, -0.20])
    ax.set_title("Adding a source graph: newcomers close the gap to best, incumbents stay near it")
    ax.grid(axis="y", ls=":", alpha=0.45)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles = [
        Line2D([], [], color=BLUE, lw=2.4, marker="o", ms=6,
               label="in-training graphs (mean gap to best)"),
        Patch(facecolor=BLUE, alpha=0.14, label=band_label),
        Line2D([], [], color=CORAL, lw=2.2, marker="o", ms=7,
               label="newly added graph: after (in-training)"),
        Line2D([], [], color=CORAL, lw=0, marker="o", mfc="white", mec=CORAL, mew=1.6,
               ms=7, label="…before it was added (out-of-dist)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8.2, framealpha=0.92,
              bbox_to_anchor=(0.005, 0.01))

    fig.tight_layout()
    fig.savefig(f"{out}.pdf")
    fig.savefig(f"{out}.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out}.pdf / .png")


def main() -> None:
    here = Path(__file__).resolve().parent
    make_figure("flat", here / "nm_ladder_gain_retention_gap")
    make_figure("std", here / "nm_ladder_gain_retention_gap_std")
    make_figure("minmax", here / "nm_ladder_gain_retention_gap_minmax")


if __name__ == "__main__":
    main()
