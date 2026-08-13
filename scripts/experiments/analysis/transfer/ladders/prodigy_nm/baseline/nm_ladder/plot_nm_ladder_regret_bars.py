#!/usr/bin/env python3
"""NM interpolation-ladder — per-step REGRET-GAP bars (role-split, + combined).

Regret framing (same definition as plot_gain_retention_gap.py): every AUC is turned
into a gap to the best model on that graph,
    GAP[r][g] = AUC(g, rung r) - BEST[g],   BEST[g] = max over all 16 models
                (8 ladder rungs + 8 single-source specialists);  GAP <= 0, 0 = best.

As the SSL pre-training merge grows one source per rung, each graph plays one of three
ROLES at the step where a new source enters (rung r, 0-indexed newcomer col = r-1):
  newcomer   : the graph just added -- was out-of-mix (deep gap), now in-mix (~0 gap).
  incumbents : graphs already in the merge (entry < r) -- "in-mix".
  held-out   : graphs not yet added   (entry > r) -- "out-of-mix".

Emits four PNGs (400 dpi, no PDF), mirroring the messages of nm_ladder_deltas.pdf:
  (a) nm_ladder_regret_newcomer_close    green bars from 0 up = the regret-gap BOOST the
                                         incoming graph gets the step it enters the merge.
  (b) nm_ladder_regret_incumbent_impact  per-step MEAN impact on the in-mix graphs' regret
                                         (Delta gap); ~0 -> adding barely hurts in-mix graphs.
  (c) nm_ladder_regret_heldout_impact    same for the out-of-mix (held-out) graphs.
  (d) nm_ladder_regret_combined          (a)+(b)+(c) stacked, shared x-axis.

Impact on regret = Delta gap = GAP[r][g] - GAP[r-1][g] (= Delta AUC, since BEST cancels):
 + = moved toward best (help), - = moved away (cost). (b)/(c) share one zoomed y-scale
so they are directly comparable; the count of graphs averaged is printed as n=. All bars
are equal width. Data embedded (mirrors nm_ladder_full.csv + nm_single_source_matrix.csv).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DPI = 400

GREEN = "#2e8b45"   # newcomer boost (out-of-mix -> in-mix)
GREEN_DK = "#1a5a2a"
CORAL, CORAL_DK = "#d85a30", "#8a3b1c"   # incumbents (in-mix)
GRAY, GRAY_DK = "#8f8d87", "#5f5e5a"     # held-out (out-of-mix)
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

BARW = 0.62         # equal width for every bar
BAND = 0.005        # ±0.5 AUC-pt "negligible" band on the impact panels

SHORT = ["ukr", "covid", "midterm", "cov_pol", "elec20", "ukr_susp", "twibot20", "cp_hk"]
# source ADDED at rungs 2..8 (rung 1 = ukr has no predecessor -> no delta).
ADDED = ["+covid", "+midterm", "+cov-pol", "+elec '20", "+ukr-susp", "+twibot", "+cp-hk"]

# rungs 1..8, AUC in canonical column order (== nm_ladder_full.csv, within_balanced).
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
# 8 single-source specialists (row = model trained on graph i), canonical columns.
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
BEST = [max(max(LADDER[r][j] for r in range(1, 9)), max(row[j] for row in SPEC))
        for j in range(8)]
GAP = {r: [LADDER[r][j] - BEST[j] for j in range(8)] for r in range(1, 9)}  # <= 0

RUNGS = list(range(2, 9))                    # steps where a source enters
X = list(range(len(RUNGS)))                  # 0..6

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
})


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def _xaxis(ax, show_x):
    ax.set_xlim(-0.6, len(X) - 0.4)
    ax.set_xticks(X)
    if show_x:
        ax.set_xticklabels(ADDED, fontsize=9.5)
        ax.set_xlabel("source added to the SSL pre-training merge", fontsize=10.5, color=INK)
    else:
        ax.tick_params(labelbottom=False)


def _save(fig, name):
    out = os.path.join(HERE, "figures", f"{name}.png")
    fig.savefig(out, bbox_inches="tight", dpi=DPI)
    print("wrote", out)
    plt.close(fig)


def _subtitle(ax, msg):
    ax.text(0.0, 1.015, "NM 3-shot / 30-way · matched step 40k · within-balanced · gap = "
            "AUC − best-per-graph over all 16 models (0 = best)\n" + msg,
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.4, color=MUTED)


# ------------------------------------------------------------------ role deltas
def role_impacts():
    """For each rung r (2..8): the per-graph Delta-gap for incumbents and held-out."""
    inc, hel = {}, {}
    for r in RUNGS:
        dgap = [GAP[r][c] - GAP[r - 1][c] for c in range(8)]   # == Delta AUC
        inc[r] = [dgap[c] for c in range(8) if c < r - 1]      # entry < r  (in-mix)
        hel[r] = [dgap[c] for c in range(8) if c > r - 1]      # entry > r  (out-of-mix)
    return inc, hel


def impact_ylim(inc, hel):
    """Shared zoomed y-scale for (b)/(c), from the bar means (+ the ±BAND band)."""
    means = [sum(per[r]) / len(per[r]) for per in (inc, hel) for r in RUNGS if per[r]]
    lo, hi = min(means + [0.0]), max(means + [0.0])
    span = hi - lo
    return (min(lo - 0.42 * span, -BAND - 0.002), max(hi + 0.40 * span, BAND + 0.002))


# ------------------------------------------------------------- reusable panels
def _newcomer_axis(ax, boost, show_x=True):
    ax.axhline(0.0, color="#c3c2b7", lw=0.9, zorder=2)
    ax.bar(X, boost, width=BARW, color=GREEN, edgecolor="white", linewidth=0.9, zorder=3)
    for xi, c in zip(X, boost):
        ax.annotate(f"+{c:.3f}", xy=(xi, c), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=GREEN_DK, fontweight="bold")
    ax.set_ylim(0, max(boost) * 1.17)
    ax.set_yticks([0.00, 0.05, 0.10, 0.15])
    _xaxis(ax, show_x)
    _style(ax)


def _impact_axis(ax, per_step, color, ann_color, ylim, show_x=True):
    """per_step[i] = list of per-graph Delta-gaps for the role at step i (may be [])."""
    ax.axhspan(-BAND, BAND, color=color, alpha=0.06, zorder=0)   # ±0.5 AUC-pt band
    ax.axhline(0.0, color="#c3c2b7", lw=1.0, zorder=2)
    for xi, vals in zip(X, per_step):
        if not vals:
            continue
        m, n = sum(vals) / len(vals), len(vals)
        ax.bar([xi], [m], width=BARW, color=color, edgecolor="white", linewidth=0.8,
               alpha=0.92, zorder=3)
        va, off = ("bottom", 4) if m >= 0 else ("top", -4)
        ax.annotate(f"{m:+.3f}", xy=(xi, m), xytext=(0, off), textcoords="offset points",
                    ha="center", va=va, fontsize=7.8, color=ann_color, fontweight="bold")
        ax.annotate(f"n={n}", xy=(xi, ylim[0]), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.4, color=MUTED)
    ax.set_ylim(*ylim)
    _xaxis(ax, show_x)
    _style(ax)


# ------------------------------------------------------------- standalone figs
def fig_newcomer(boost):
    fig, ax = plt.subplots(figsize=(9.0, 4.9), dpi=DPI, constrained_layout=True)
    _newcomer_axis(ax, boost)
    ax.set_ylabel("regret-gap boost on the newcomer\n(AUC pts the gap closes)",
                  fontsize=10, color=INK)
    ax.set_title("Adding a graph closes its own regret gap — the boost on the incoming graph",
                 fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=24)
    _subtitle(ax, r"each bar = how far the incoming graph's gap to best shrinks the step it "
                  r"enters the merge (out-of-mix $\rightarrow$ in-mix)")
    _save(fig, "nm_ladder_regret_newcomer_close")


def fig_incumbent(inc, ylim):
    fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=DPI, constrained_layout=True)
    _impact_axis(ax, [inc[r] for r in RUNGS], CORAL, CORAL_DK, ylim)
    ax.set_ylabel("impact on in-mix regret\n(Δ gap; + = toward best)", fontsize=10, color=INK)
    ax.set_title("Adding a graph barely moves the in-mix graphs' regret",
                 fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=24)
    _subtitle(ax, "each bar = mean Δ gap over the graphs already in the merge (count = n) · "
                  "shaded band = ±0.5 AUC pts (negligible)")
    _save(fig, "nm_ladder_regret_incumbent_impact")


def fig_heldout(hel, ylim):
    fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=DPI, constrained_layout=True)
    _impact_axis(ax, [hel[r] for r in RUNGS], GRAY, GRAY_DK, ylim)
    ax.set_ylabel("impact on out-of-mix regret\n(Δ gap; + = toward best)", fontsize=10, color=INK)
    ax.annotate("none held out", xy=(X[-1], 0), xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.8, color=MUTED, style="italic")
    ax.set_title("Adding a graph barely moves the out-of-mix graphs' regret",
                 fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=24)
    _subtitle(ax, "each bar = mean Δ gap over the graphs not yet in the merge (count = n) · "
                  "shaded band = ±0.5 AUC pts (negligible)")
    _save(fig, "nm_ladder_regret_heldout_impact")


# --------------------------------------------------------------- combined fig
def fig_combined(inc, hel, boost):
    """One axes, one scale: the three roles' Δ regret gap grouped per step. The
    newcomer's Δ gap (its boost) towers; the in-mix / out-of-mix means hug zero."""
    inc_m = [sum(inc[r]) / len(inc[r]) if inc[r] else None for r in RUNGS]
    hel_m = [sum(hel[r]) / len(hel[r]) if hel[r] else None for r in RUNGS]

    w = 0.27
    xN = [i - w for i in X]
    xI = list(X)
    xO = [i + w for i in X]

    fig, ax = plt.subplots(figsize=(10.6, 5.7), dpi=DPI, constrained_layout=True)
    ax.axhline(0.0, color="#9c9a93", lw=1.0, zorder=2)

    ax.bar(xN, boost, width=w, color=GREEN, edgecolor="white", linewidth=0.7, zorder=3,
           label="newcomer — the graph just added")
    ax.bar(xI, [m or 0.0 for m in inc_m], width=w, color=CORAL, edgecolor="white",
           linewidth=0.7, zorder=3, label="in-mix graphs — mean (already in the merge)")
    xO2 = [x for x, m in zip(xO, hel_m) if m is not None]
    ax.bar(xO2, [m for m in hel_m if m is not None], width=w, color=GRAY, edgecolor="white",
           linewidth=0.7, zorder=3, label="out-of-mix graphs — mean (not yet in the merge)")

    for x, c in zip(xN, boost):
        ax.annotate(f"+{c:.3f}", xy=(x, c), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.4, color=GREEN_DK, fontweight="bold")

    ax.set_ylim(-0.014, max(boost) * 1.12)
    ax.set_yticks([0.00, 0.05, 0.10, 0.15])
    ax.set_ylabel("Δ regret gap   (AUC pts; + = toward best)", fontsize=10, color=INK)
    ax.set_xlim(-0.6, len(X) - 0.4)
    ax.set_xticks(X)
    ax.set_xticklabels(ADDED, fontsize=9.5)
    ax.set_xlabel("source added to the SSL pre-training merge", fontsize=10.5, color=INK)
    _style(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9, borderaxespad=0.4,
              handlelength=1.2, labelspacing=0.45)
    ax.set_title("Adding a source: big boost on the newcomer, ~zero impact on everyone else",
                 fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=24)
    _subtitle(ax, "every bar = Δ regret gap at that step on one shared scale · newcomer = the "
                  "added graph; the in-mix / out-of-mix means stay within ±0.006 of zero")
    _save(fig, "nm_ladder_regret_combined")


def main():
    inc, hel = role_impacts()
    before = [GAP[r - 1][r - 1] for r in RUNGS]
    after = [GAP[r][r - 1] for r in RUNGS]
    boost = [a - b for a, b in zip(after, before)]
    ylim = impact_ylim(inc, hel)

    fig_newcomer(boost)
    fig_incumbent(inc, ylim)
    fig_heldout(hel, ylim)
    fig_combined(inc, hel, boost)

    print("\n(a) newcomer — regret gap before → after (boost = gap closed)")
    for a, b, af, c in zip(ADDED, before, after, boost):
        print(f"  {a:<10} {b:+.4f} → {af:+.4f}   boost +{c:.4f}")
    print(f"\n(b)/(c) mean Δ-gap impact  (shared y-scale [{ylim[0]:+.4f}, {ylim[1]:+.4f}])")
    print("  step        in-mix (n)             out-of-mix (n)")
    for i, r in enumerate(RUNGS):
        iv, hv = inc[r], hel[r]
        im = f"{sum(iv)/len(iv):+.4f} (n={len(iv)})" if iv else "      —       "
        hm = f"{sum(hv)/len(hv):+.4f} (n={len(hv)})" if hv else "   — none —"
        print(f"  {ADDED[i]:<10} {im:<22} {hm}")


if __name__ == "__main__":
    main()
