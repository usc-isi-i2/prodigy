#!/usr/bin/env python3
"""Does graph similarity account for the ladder's transfer? Two panels.

Panel A — mechanism: each held-out graph's OBSERVED out-of-distribution transfer in
the ladder (its AUC the rung before it enters) vs. the BEST single-source donor
already in the merge (max over merge members of the single-source NM transfer to that
graph). They lie on the identity line — the merged model transfers to a new graph
about as well as its single most-compatible member already did (Spearman ρ = 1.00;
observed ≈ best − 0.005, a small shared-budget dilution). So each source's entry
impact = its own in-domain ceiling minus how well the pre-existing best match covered it.

Panel B — the "why": that best-match quality is a graph-similarity quantity. Observed
OOD vs. the graph's divergence to the merge on both axes — FEATURE (min proxy-A
distance) and TOPOLOGY (min in-degree KS). Both trend down (more divergent ⇒ lower
transfer), but n = 7 so the axes are not cleanly separable here (the feature-vs-topology
separation is the single-source-matrix result). The one qualitative split: cp_hk is
topology-CLOSE to the merge yet transfers WORST — topology mispredicts it, feature does
not (the known cp_hk topology anomaly).

Derived (see below) from:
  - scripts/experiments/analysis/nm_ladder/data/nm_ladder_full.csv              (ladder OOD baselines)
  - scripts/experiments/analysis/nm_single_source_matrix/data/nm_single_source_matrix.csv  (donors)
  - scripts/experiments/analysis/graph_divergence/data/graph_divergence_data.json (pairwise divergence)
Values are embedded so the figure is self-contained (the inputs are gitignored / churned
by a parallel analysis). Recompute: rerun the join over the three files above.

Writes nm_ladder_vs_similarity.pdf/.png here.
"""
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))

BLUE = "#2a78d6"    # feature axis
CORAL = "#d85a30"   # topology axis
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

# per added graph (entering rungs 2..8): observed OOD baseline, best single-source
# donor in the merge, min feature-divergence (proxy-A) to merge, min topology-
# divergence (in-degree KS) to merge.  (ukr = rung 1, always in-training, excluded.)
DATA = [
    # short,     ood,   best_donor, feat_div, topo_div
    ("covid",    0.973, 0.973, 0.193, 0.022),
    ("midterm",  0.885, 0.884, 0.810, 0.037),
    ("cov_pol",  0.830, 0.850, 1.680, 0.456),
    ("elec20",   0.830, 0.835, 0.772, 0.800),
    ("ukr_susp", 0.769, 0.789, 0.642, 0.176),
    ("twibot",   0.924, 0.926, 0.592, 0.081),
    ("cp_hk",    0.727, 0.720, 1.015, 0.083),
]


def rankdata(a):
    order = sorted(range(len(a)), key=lambda i: a[i])
    rk = [0.0] * len(a)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        for k in range(i, j + 1):
            rk[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return rk


def spearman(x, y):
    x, y = rankdata(x), rankdata(y)
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else float("nan")


def polyfit1(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    m = sxy / sxx
    return m, my - m * mx


def main():
    short = [d[0] for d in DATA]
    ood = [d[1] for d in DATA]
    best = [d[2] for d in DATA]
    fdiv = [d[3] for d in DATA]
    tdiv = [d[4] for d in DATA]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
    })
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.5), dpi=200)

    # ---------------- Panel A: observed OOD vs best single-source donor ----------------
    lo, hi = 0.70, 1.0
    axA.plot([lo, hi], [lo, hi], color="#c3c2b7", lw=1.2, ls=(0, (4, 2)), zorder=1)
    axA.annotate("identity", xy=(0.815, 0.815), xytext=(2, -13), textcoords="offset points",
                 ha="center", va="top", fontsize=8.5, color=MUTED, rotation=45)
    sc = axA.scatter(best, ood, c=fdiv, cmap="viridis_r", s=150, edgecolor="white",
                     linewidth=1.0, zorder=4)
    LPOS = {  # dx, dy (points), ha, va — keep labels off each other + in frame
        "covid":    (-7, -3, "right", "top"),
        "twibot":   (7, -2, "left", "top"),
        "midterm":  (7, -1, "left", "top"),
        "cov_pol":  (7, 3, "left", "bottom"),
        "elec20":   (-7, -3, "right", "top"),
        "ukr_susp": (7, -2, "left", "top"),
        "cp_hk":    (7, -2, "left", "top"),
    }
    for s, bx, oy in zip(short, best, ood):
        dx, dy, ha, va = LPOS.get(s, (6, -3, "left", "top"))
        axA.annotate(s, xy=(bx, oy), xytext=(dx, dy), textcoords="offset points",
                     ha=ha, va=va, fontsize=9, color=INK)
    rhoA = spearman(best, ood)
    gap = sum(o - b for o, b in zip(ood, best)) / len(ood)
    axA.text(0.02, 0.97, f"Spearman ρ = {rhoA:+.2f}\nobserved ≈ best − {abs(gap):.3f}",
             transform=axA.transAxes, ha="left", va="top", fontsize=10, color=INK,
             fontweight="bold")
    cb = fig.colorbar(sc, ax=axA, fraction=0.046, pad=0.03)
    cb.set_label("feature divergence to merge\n(min proxy-A distance)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    axA.set_xlim(lo, hi)
    axA.set_ylim(lo, hi)
    axA.set_xlabel("best single-source donor in the merge  (expected OOD)", fontsize=10.5, color=INK)
    axA.set_ylabel("observed ladder OOD transfer  (AUC before entry)", fontsize=10.5, color=INK)
    axA.set_title("The merge transfers like its best-matching source",
                  fontsize=11.5, color=INK, fontweight="bold", loc="left", pad=10)
    axA.tick_params(colors=MUTED, labelsize=9)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axA.spines[sp].set_color("#c3c2b7")
    axA.grid(color=GRID, lw=0.8, zorder=0)
    axA.set_axisbelow(True)

    # ---------------- Panel B: OOD vs feature / topology divergence (normalized) ----------------
    def norm(v):
        lo_, hi_ = min(v), max(v)
        return [(x - lo_) / (hi_ - lo_) for x in v]

    fx, tx = norm(fdiv), norm(tdiv)
    rho_f, rho_t = spearman(fdiv, ood), spearman(tdiv, ood)
    xl = [-0.02, 1.02]
    mf, bf = polyfit1(fx, ood)
    mt, bt = polyfit1(tx, ood)
    axB.plot(xl, [mf * x + bf for x in xl], color=BLUE, lw=1.6, ls="-", zorder=2, alpha=0.7)
    axB.plot(xl, [mt * x + bt for x in xl], color=CORAL, lw=1.6, ls="-", zorder=2, alpha=0.7)
    axB.scatter(fx, ood, color=BLUE, marker="o", s=95, edgecolor="white", linewidth=1.0, zorder=4)
    axB.scatter(tx, ood, color=CORAL, marker="s", s=85, edgecolor="white", linewidth=1.0, zorder=4)
    # highlight the cp_hk topology anomaly (topology-close, worst transfer)
    ci = short.index("cp_hk")
    axB.annotate("cp_hk: topology-close\nbut transfers worst\n(topology mispredicts)",
                 xy=(tx[ci], ood[ci]), xytext=(0.30, 0.745), fontsize=8.4, color="#8a3b1c",
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.0))
    axB.set_xlim(*xl)
    axB.set_ylim(lo, hi)
    axB.set_xlabel("relative divergence to merge  (0 = nearest member, 1 = farthest)",
                   fontsize=10.5, color=INK)
    axB.set_ylabel("observed ladder OOD transfer", fontsize=10.5, color=INK)
    axB.set_title("Transfer falls with divergence — feature vs topology  (n = 7)",
                  fontsize=11.5, color=INK, fontweight="bold", loc="left", pad=10)
    axB.tick_params(colors=MUTED, labelsize=9)
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axB.spines[sp].set_color("#c3c2b7")
    axB.grid(color=GRID, lw=0.8, zorder=0)
    axB.set_axisbelow(True)
    handles = [
        Line2D([0], [0], color=BLUE, lw=1.6, marker="o", markerfacecolor=BLUE,
               markeredgecolor="white", ms=8, label=f"feature (proxy-A)   ρ = {rho_f:+.2f}"),
        Line2D([0], [0], color=CORAL, lw=1.6, marker="s", markerfacecolor=CORAL,
               markeredgecolor="white", ms=7, label=f"topology (in-deg KS)   ρ = {rho_t:+.2f}"),
    ]
    axB.legend(handles=handles, loc="upper right", frameon=False, fontsize=9,
               handlelength=2.0, borderaxespad=0.6)

    fig.suptitle("Graph similarity accounts for the ladder's out-of-distribution transfer",
                 fontsize=13, color=INK, fontweight="bold", x=0.012, ha="left", y=1.005)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("pdf", "png"):
        out = os.path.join(HERE, f"nm_ladder_vs_similarity.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)

    print(f"\nPanel A: rho(OOD, best-donor) = {rhoA:+.3f} ; observed − best = {gap:+.3f}")
    print(f"Panel B: rho(OOD, feature-div) = {rho_f:+.2f} ; rho(OOD, topo-div) = {rho_t:+.2f}")


if __name__ == "__main__":
    main()
