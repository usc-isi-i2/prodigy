#!/usr/bin/env python3
"""NM interpolation-ladder — per-source impact (role-decomposed deltas).

For each source added to the merge (rungs 2..8), split the per-graph AUC change
Delta(g) = AUC(g, this rung) - AUC(g, rung below) by the graph's ROLE at that step:
  - newcomer   : the graph just added -- its own out->in-distribution jump (BENEFIT).
  - incumbents : graphs already in training -- mean Delta (COST / dilution).
  - held-out   : graphs not yet added   -- mean Delta (cross-source interference).

Averaging *within* a role is the right "mean of deltas"; averaging *across* roles
(or taking the delta of the overall mean) dilutes the newcomer's jump ~8x and can
even flip its sign. The newcomer benefit is large (+.01..+.17); the incumbent cost
and held-out interference are ~0 -- so the two panels use different y-scales (benefit
at full scale on top, cost/interference zoomed on the bottom).

Shares data with the other nm_ladder plots (nm_ladder_full.csv within_balanced rows;
embedded fallback since the CSV is gitignored). Writes nm_ladder_deltas.pdf/.png here.
"""
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))

BLUE = "#2a78d6"    # newcomer benefit (OOD->ID)
CORAL = "#d85a30"   # incumbents (cost / dilution)
GRAY = "#8f8d87"    # held-out (interference)
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

CANON = [
    "ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
    "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter",
]
ENTRY = {k: i + 1 for i, k in enumerate(CANON)}  # one graph enters per rung
# x labels = the source ADDED at rungs 2..8 (rung 1 = ukr has no delta).
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
    for r in range(2, 9):                      # rung r; index r-1
        d = {c: series[c][r - 1] - series[c][r - 2] for c in CANON}
        newcomer = CANON[r - 1]                # graph entering at rung r
        incs = [c for c in CANON if ENTRY[c] < r]
        hels = [c for c in CANON if ENTRY[c] > r]
        newc.append(d[newcomer])
        inc.append(sum(d[c] for c in incs) / len(incs))
        hel.append(sum(d[c] for c in hels) / len(hels) if hels else math.nan)

    x = list(range(len(ADDED)))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
    })
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.4, 6.6), dpi=200, sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [2.4, 1.0]})

    # -- top: newcomer benefit (full scale) --
    ax1.bar(x, newc, width=0.62, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
    for xi, v in zip(x, newc):
        ax1.annotate(f"+{v:.3f}", xy=(xi, v), xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, color=BLUE, fontweight="bold")
    ax1.set_ylim(0, 0.185)
    ax1.set_yticks([0.00, 0.05, 0.10, 0.15])
    ax1.set_ylabel("newcomer Δ\n(OOD to ID jump)", fontsize=10, color=INK)
    ax1.axhline(0, color="#c3c2b7", lw=0.9, zorder=2)
    ax1.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax1.set_axisbelow(True)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax1.spines[sp].set_color("#c3c2b7")
    ax1.tick_params(colors=MUTED, labelsize=9)

    # -- bottom: incumbents + held-out (zoomed) --
    w = 0.36
    xi_inc = [xi - w / 2 for xi in x]
    xi_hel = [xi + w / 2 for xi in x]
    ax2.bar(xi_inc, inc, width=w, color=CORAL, edgecolor="white", linewidth=0.8, zorder=3)
    ax2.bar(xi_hel, hel, width=w, color=GRAY, edgecolor="white", linewidth=0.8, zorder=3)
    for xi, v in zip(xi_inc, inc):
        off = 4 if v >= 0 else -4
        va = "bottom" if v >= 0 else "top"
        ax2.annotate(f"{v:+.3f}", xy=(xi, v), xytext=(0, off), textcoords="offset points",
                     ha="center", va=va, fontsize=7.2, color="#8a3b1c")
    for xi, v in zip(xi_hel, hel):
        if math.isnan(v):
            continue
        off = 4 if v >= 0 else -4
        va = "bottom" if v >= 0 else "top"
        ax2.annotate(f"{v:+.3f}", xy=(xi, v), xytext=(0, off), textcoords="offset points",
                     ha="center", va=va, fontsize=7.2, color="#5f5e5a")
    ax2.axhline(0, color="#c3c2b7", lw=0.9, zorder=2)
    ax2.set_ylim(-0.010, 0.008)
    ax2.set_yticks([-0.01, 0.00, 0.005])
    ax2.set_ylabel("mean Δ of\nother graphs", fontsize=10, color=INK)
    ax2.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax2.set_axisbelow(True)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax2.spines[sp].set_color("#c3c2b7")
    ax2.tick_params(colors=MUTED, labelsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ADDED, fontsize=9.5)
    ax2.set_xlabel("source added to the SSL pre-training merge", fontsize=10.5, color=INK)

    # 3-role legend in the empty top-left of the benefit panel
    handles = [
        Patch(facecolor=BLUE, label="newcomer (just added) — benefit"),
        Patch(facecolor=CORAL, label="incumbents (already in training) — cost"),
        Patch(facecolor=GRAY, label="held-out (not yet added) — interference"),
    ]
    ax1.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.8,
               handlelength=1.2, borderaxespad=0.6, labelspacing=0.5)

    ax1.set_title("Per-source impact: large in-domain benefit, ~zero cost to the other graphs",
                  fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=24)
    ax1.text(0.0, 1.03, "NM 3-shot / 30-way · matched step 40k · within-balanced "
             "· Δ = AUC gain at this rung vs the rung below  (note: panels differ in y-scale)",
             transform=ax1.transAxes, ha="left", va="bottom", fontsize=8.8, color=MUTED)

    for ext in ("pdf", "png"):
        out = os.path.join(HERE, f"nm_ladder_deltas.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print("wrote", out)

    print("\nadd        newcomer   incumbents   held-out")
    for i, a in enumerate(ADDED):
        h = f"{hel[i]:+.4f}" if not math.isnan(hel[i]) else "   -"
        print(f"  {a:<10}{newc[i]:+.4f}    {inc[i]:+.4f}    {h}")


if __name__ == "__main__":
    main()
