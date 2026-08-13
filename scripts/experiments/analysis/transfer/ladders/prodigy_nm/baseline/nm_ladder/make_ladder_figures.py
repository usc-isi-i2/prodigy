#!/usr/bin/env python3
"""NM graph-ladder figures unique to this chat session, as paper-ready PDF+PNG in
./figures/. Self-contained: the 16x8 AUC matrix (8 merged ladder rungs + 8
single-source specialists, NM 3-shot/30-way, matched-40k, within-balanced) is
embedded below — identical to nm_ladder_full.csv + nm_single_source_matrix.csv.

NOTE: the combined 16-row heatmap + the regret heatmap are ALSO produced by the
sibling plot_heatmaps.py (a parallel chat's script) as
nm_ladder_plus_single_source_heatmap.pdf / nm_ladder_regret_heatmap.pdf, and the base
generalist scatter by plot_generalist_scatter.py. To avoid clobbering those, this
script emits only the two figures unique to this session:
  nm_ladder_heatmap                        8-rung ladder-only AUC heatmap (staircase)
  nm_ladder_generalist_scatter_errorbars   scatter WITH +/-1 s.e.m. error bars
The fig_combined_heatmap()/fig_regret() functions below are kept for reference but are
not called from __main__.
"""
import os
import statistics as st
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
CORAL = "#D85A30"
SHORT = ["ukr", "covid", "midterm", "cov_pol", "elec20", "ukr_susp", "twibot", "cp_hk"]

# rows: (label, 8 AUCs) in CANON column order
LADDER = [
    ("L1 ukr",       [.9480, .9730, .8740, .8490, .8280, .7710, .9210, .7240]),
    ("L2 +cov",      [.9450, .9800, .8850, .8430, .8280, .7750, .9250, .7260]),
    ("L3 +mid",      [.9410, .9780, .9150, .8300, .8150, .7770, .9270, .7200]),
    ("L4 +cov_pol",  [.9344, .9753, .9093, .9113, .8297, .7768, .9234, .7235]),
    ("L5 +elec20",   [.9346, .9754, .9086, .9102, .9259, .7693, .9254, .7261]),
    ("L6 +ukr_susp", [.9325, .9744, .9073, .9106, .9241, .9340, .9242, .7239]),
    ("L7 +twibot20", [.9321, .9748, .9033, .9076, .9198, .9256, .9377, .7267]),
    ("L8 all8",      [.9340, .9750, .9080, .9060, .9200, .9310, .9370, .8670]),
]
SINGLE = [
    ("ukr",       [.9470, .9730, .8811, .8394, .8262, .7894, .9218, .7140]),
    ("covid",     [.9264, .9805, .8842, .8498, .8345, .7861, .9258, .7196]),
    ("midterm",   [.7970, .8790, .9252, .8353, .8047, .6440, .8613, .6256]),
    ("cov_pol",   [.6301, .6985, .6878, .9145, .7833, .5510, .7374, .5444]),
    ("elec20",    [.6018, .6551, .6804, .7872, .9519, .5627, .7096, .5475]),
    ("ukr_susp",  [.7695, .8307, .7247, .7325, .7275, .9640, .7647, .6225]),
    ("twibot20",  [.8688, .9473, .8600, .8431, .8025, .7115, .9487, .6895]),
    ("cp_hk",     [.6808, .7579, .7629, .6827, .6412, .6025, .7378, .9055]),
]

BLUES = LinearSegmentedColormap.from_list("nm_blue", ["#E6F1FB", "#85B7EB", "#185FA5", "#0C447C"])
REDS = LinearSegmentedColormap.from_list("nm_red", ["#FCEBEB", "#F09595", "#A32D2D", "#791F1F"])
GREENS = LinearSegmentedColormap.from_list("nm_green", [(181/255, 224/255, 134/255), (40/255, 86/255, 14/255)])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
})


def _txt_color(t):
    return "white" if t > 0.55 else "#20303f"


def _save(fig, name):
    for ext in ("pdf", "png"):
        out = os.path.join(FIGDIR, f"{name}.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)
    plt.close(fig)


def _heat_cells(ax, rows, entry_of_row, vmin=0.68, vmax=0.985):
    """Draw an AUC heatmap on ax. rows = [(label, [8 vals])]; entry_of_row[i] = the
    col index that first enters at row i (marked), or None."""
    n = len(rows)
    for i, (_, vals) in enumerate(rows):
        for j, v in enumerate(vals):
            t = (v - vmin) / (vmax - vmin)
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, facecolor=BLUES(max(0, min(1, t))),
                                   edgecolor="white", linewidth=1.2))
            ax.text(j + 0.5, n - 1 - i + 0.5, f"{v:.3f}"[1:], ha="center", va="center",
                    fontsize=7.6, color=_txt_color(t))
        e = entry_of_row[i]
        if e is not None:
            ax.add_patch(Rectangle((e, n - 1 - i), 1, 1, fill=False,
                                   edgecolor="#D85A30", linewidth=2.2, zorder=5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, n)
    ax.set_xticks([j + 0.5 for j in range(8)])
    ax.set_xticklabels(SHORT, fontsize=8.5, rotation=35, ha="right")
    ax.set_yticks([n - 1 - i + 0.5 for i in range(n)])
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")


def fig_ladder_heatmap():
    fig, ax = plt.subplots(figsize=(7.4, 6.0), dpi=200)
    _heat_cells(ax, LADDER, entry_of_row=list(range(8)))
    ax.set_title("NM interpolation ladder — each column jumps when its graph enters training",
                 fontsize=11.5, color=INK, fontweight="bold", loc="left", pad=14)
    ax.text(0, 1.012, "test AUC (NM 3-shot / 30-way, matched-40k) · orange box = the graph that "
            "first enters the merge at that rung", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.3, color=MUTED)
    fig.tight_layout()
    _save(fig, "nm_ladder_heatmap")


def fig_combined_heatmap():
    fig, ax = plt.subplots(figsize=(7.6, 9.2), dpi=200)
    rows = LADDER + SINGLE
    entry = list(range(8)) + list(range(8))  # ladder diagonal + specialist diagonal
    _heat_cells(ax, rows, entry_of_row=entry)
    ax.axhline(8, color=INK, lw=1.4)  # divider between blocks (rows: single at bottom 0..7)
    ax.text(-0.6, 12, "merged\nladder", ha="right", va="center", fontsize=8.4,
            color=MUTED, fontweight="bold", rotation=0)
    ax.text(-0.6, 4, "single-\nsource", ha="right", va="center", fontsize=8.4,
            color=MUTED, fontweight="bold", rotation=0)
    ax.set_title("Ladder + single-source specialists — 16 models on 8 test graphs",
                 fontsize=11.5, color=INK, fontweight="bold", loc="left", pad=14)
    ax.text(0, 1.008, "test AUC (NM 3-shot / 30-way, matched-40k) · orange box = graph enters "
            "merge (ladder) / in-domain specialist (single-source)", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=8.0, color=MUTED)
    fig.tight_layout()
    _save(fig, "nm_ladder_plus_single_source_heatmap")


def _rank_col(mat, j):
    vals = [r[j] for r in mat]
    return [sum(1 for x in vals if x > vals[i]) + 1 + (sum(1 for x in vals if x == vals[i]) - 1) / 2
            for i in range(len(vals))]


def fig_regret():
    rows = [(r[0], r[1]) for r in LADDER] + [("S " + n, v) for n, v in SINGLE]
    mat = [r[1] for r in rows]
    labels = [r[0] for r in rows]
    best = [max(r[j] for r in mat) for j in range(8)]
    delta = [[(mat[i][j] - best[j]) * 100 for j in range(8)] for i in range(len(mat))]
    ranks = [_rank_col(mat, j) for j in range(8)]
    mean_d = [sum(delta[i]) / 8 for i in range(len(mat))]
    mean_r = [sum(ranks[j][i] for j in range(8)) / 8 for i in range(len(mat))]
    n = len(mat)

    fig, (axm, axs) = plt.subplots(
        1, 2, figsize=(9.6, 8.6), dpi=200,
        gridspec_kw={"width_ratios": [8, 2], "wspace": 0.06})

    for i in range(n):
        for j in range(8):
            m = -delta[i][j]
            if m < 0.05:
                axm.add_patch(Rectangle((j, n - 1 - i), 1, 1, facecolor="#E1F5EE",
                                        edgecolor="#0F6E56", linewidth=2.0))
                axm.text(j + 0.5, n - 1 - i + 0.5, "0", ha="center", va="center",
                         fontsize=7.4, color="#0F6E56", fontweight="bold")
            else:
                t = min(1, m / 30)
                axm.add_patch(Rectangle((j, n - 1 - i), 1, 1, facecolor=REDS(t),
                                        edgecolor="white", linewidth=1.2))
                axm.text(j + 0.5, n - 1 - i + 0.5, f"−{m:.1f}", ha="center", va="center",
                         fontsize=7.0, color=_txt_color(t))
    axm.axhline(8, color=INK, lw=1.4)
    axm.set_xlim(0, 8); axm.set_ylim(0, n)
    axm.set_xticks([j + 0.5 for j in range(8)]); axm.set_xticklabels(SHORT, fontsize=8.5, rotation=35, ha="right")
    axm.set_yticks([n - 1 - i + 0.5 for i in range(n)]); axm.set_yticklabels(labels, fontsize=8.3)
    axm.tick_params(length=0)
    for s in axm.spines.values():
        s.set_visible(False)
    axm.set_aspect("equal")

    # summary columns: mean delta, mean rank
    for i in range(n):
        md = -mean_d[i]
        axs.add_patch(Rectangle((0, n - 1 - i), 1, 1, facecolor=REDS(min(1, md / 26)),
                                edgecolor="white", linewidth=1.2))
        axs.text(0.5, n - 1 - i + 0.5, f"−{md:.1f}", ha="center", va="center",
                 fontsize=7.2, color=_txt_color(min(1, md / 26)), fontweight="bold")
        mr = mean_r[i]
        tr = (mr - 4) / 10
        axs.add_patch(Rectangle((1, n - 1 - i), 1, 1, facecolor=REDS(max(0, min(1, tr))),
                                edgecolor="white", linewidth=1.2))
        axs.text(1.5, n - 1 - i + 0.5, f"{mr:.1f}", ha="center", va="center",
                 fontsize=7.2, color=_txt_color(max(0, min(1, tr))), fontweight="bold")
    axs.axhline(8, color=INK, lw=1.4)
    axs.set_xlim(0, 2); axs.set_ylim(0, n)
    axs.set_xticks([0.5, 1.5]); axs.set_xticklabels(["mean Δ", "mean rank"], fontsize=8.4)
    axs.set_yticks([]); axs.tick_params(length=0)
    for s in axs.spines.values():
        s.set_visible(False)
    axs.set_aspect("equal")

    axm.set_title("Regret — gap in AUC points to the best model on each graph",
                  fontsize=11.5, color=INK, fontweight="bold", loc="left", pad=14)
    axm.text(0, 1.01, "0 (green) = best on that graph · darker red = further below · mean Δ and "
             "mean rank summarize each model as a generalist", transform=axm.transAxes,
             ha="left", va="bottom", fontsize=8.0, color=MUTED)
    fig.tight_layout()
    _save(fig, "nm_ladder_regret")


def fig_scatter():
    # home-turf (in-domain) x, breadth y, +/- 1 s.e.m. across the averaged graphs
    def breadth_ysem(vals):
        return sum(vals) / 8, st.pstdev(vals) / math.sqrt(8)

    pts = []  # (label, home, breadth, xsem, ysem, color, rung)
    for rung, (_, vals) in enumerate(LADDER, start=1):
        b, ys = breadth_ysem(vals)
        tr = vals[:rung]                       # graphs the rung trained on
        home = sum(tr) / rung
        xs = st.pstdev(tr) / math.sqrt(rung) if rung >= 2 else 0.0
        pts.append((f"L{rung}", home, b, xs, ys, GREENS((rung - 1) / 7), rung))
    for idx, (name, vals) in enumerate(SINGLE):
        b, ys = breadth_ysem(vals)
        pts.append((name, vals[idx], b, 0.0, ys, CORAL, 0))  # home-turf = diagonal

    fig, ax = plt.subplots(figsize=(7.8, 7.0), dpi=200)
    lo, hi = 0.62, 0.99
    ax.plot([lo, hi], [lo, hi], ls=(0, (5, 4)), color=MUTED, lw=1.4, zorder=1)
    ax.text(0.685, 0.688, "y = x  (perfect generalist)", rotation=45, rotation_mode="anchor",
            ha="left", va="bottom", fontsize=8.6, color=MUTED)
    # ladder path
    lad = sorted([p for p in pts if p[6] > 0], key=lambda p: p[6])
    ax.plot([p[1] for p in lad], [p[2] for p in lad], color="#5b9a2f", lw=1.8, alpha=0.55, zorder=2)
    for lb, h, b, xs, ys, c, rg in pts:
        ax.errorbar(h, b, xerr=(xs if xs > 0 else None), yerr=ys, fmt="none",
                    ecolor=c, elinewidth=1.1, capsize=2.4, alpha=0.55, zorder=3)
        ax.scatter([h], [b], s=(120 if rg == 8 else 46), color=c, zorder=4,
                   edgecolor=("white" if rg == 8 else "#3a3a3a"),
                   linewidth=(1.4 if rg == 8 else 0.4))
    # labels: only L8 (colorbar carries L1-L7) + the specialists
    OFF = {"ukr": (-6, -12, "right"), "covid": (-8, 3, "right"), "midterm": (7, 2, "left"),
           "cov_pol": (7, 2, "left"), "elec20": (7, -3, "left"), "ukr_susp": (7, 2, "left"),
           "twibot20": (7, 2, "left"), "cp_hk": (7, 2, "left")}
    for lb, h, b, xs, ys, c, rg in pts:
        if rg == 8:
            ax.annotate("L8·all8", (h, b), textcoords="offset points", xytext=(9, 5),
                        fontsize=8.4, color="#173404", ha="left", fontweight="bold")
        elif rg == 0:
            dx, dy, ha = OFF[lb]
            ax.annotate(lb, (h, b), textcoords="offset points", xytext=(dx, dy),
                        fontsize=8.0, color="#712b13", ha=ha)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("home-turf AUC  (mean over the graphs the model trained on)", fontsize=10, color=INK)
    ax.set_ylabel("breadth  (mean AUC over all 8 graphs)", fontsize=10, color=INK)
    ax.set_xticks([0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    ax.set_yticks([0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    ax.tick_params(colors=MUTED, labelsize=8.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.grid(color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
    sm = plt.cm.ScalarMappable(cmap=GREENS)
    cb = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.02, ticks=[0, 1])
    cb.ax.set_yticklabels(["L1", "L8"], fontsize=8)
    cb.set_label("ladder rung", fontsize=8.4, color=MUTED)
    cb.outline.set_visible(False)
    ax.scatter([], [], s=46, color=CORAL, label="single-source specialist")
    ax.legend(loc="lower right", frameon=False, fontsize=8.6)
    ax.set_title("Generalist vs specialist — the merged model reaches the frontier",
                 fontsize=11.5, color=INK, fontweight="bold", loc="left", pad=22)
    ax.text(0, 1.02, "equal axes · distance below y=x = generalization gap · bars = ±1 s.e.m. "
            "across graphs (single seed)", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.4, color=MUTED)
    _save(fig, "nm_ladder_generalist_scatter_errorbars")


if __name__ == "__main__":
    # only the figures unique to this session (see module docstring); the combined
    # heatmap + regret heatmap are the parallel chat's plot_heatmaps.py.
    fig_ladder_heatmap()
    fig_scatter()
    print("\nfigures written to", FIGDIR)
