#!/usr/bin/env python3
"""Order-robustness versions of the combined regret-bar figure (nm_ladder_regret_combined).

The original figure (nm_ladder/plot_nm_ladder_regret_bars.py, fig_combined) shows, for
every source-addition step of the published ladder, three bars on one shared scale:
  newcomer    (green) the Delta regret gap of the graph just added = its entry boost
  in-mix      (coral) mean Delta gap of the graphs already in the merge
  out-of-mix  (gray)  mean Delta gap of the graphs not yet added
Delta regret gap between consecutive rungs == Delta AUC (the best-per-graph reference
cancels), so everything here is computed straight from the order-robustness table.

This script emits the same figure for each of the three orders (A published, B strongest
donors first, C the reverse) plus one order-aggregated version:
  order_regret_combined_A / _B / _C   per-order, x = the source added at that step
  order_regret_combined_minmax       x = merge size after the addition (the newcomer
                                      differs by order); bar = mean over orders A/B/C,
                                      whisker = min/max over the three orders
All four share one y-scale so the orders are directly comparable (C's entry jumps are
~2x A's / B's: same mechanism, weaker prior coverage). Note order C also breaks the
"~zero impact on everyone else" headline: its held-out graphs recover +.01-.07 per step
(headroom recovery, see order_heldout_headroom), and the titles say so.

Reads data/nm_ladder_order_robustness_long.csv; writes figures/. Local homebrew python:
  /opt/homebrew/bin/python3.11 plot_regret_combined_by_order.py
"""
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "nm_ladder_order_robustness_long.csv")
FIGS = os.path.join(HERE, "figures")

# palette == nm_ladder/plot_nm_ladder_regret_bars.py
GREEN, GREEN_DK = "#2e8b45", "#1a5a2a"   # newcomer boost
CORAL, CORAL_DK = "#d85a30", "#8a3b1c"   # in-mix (incumbents)
GRAY, GRAY_DK = "#8f8d87", "#5f5e5a"     # out-of-mix (held-out)
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

STEPS = list(range(2, 9))                # rung r vs r-1: the 7 addition steps
X = list(range(len(STEPS)))
ORDERS = ("A", "B", "C")
ORDER_DESC = {"A": "published topical order", "B": "strongest donors first",
              "C": "weakest donors first"}
LABEL = {"ukr_rus": "+ukr", "covid": "+covid", "midterm": "+midterm",
         "covid_political": "+cov-pol", "election2020": "+elec '20",
         "ukr_rus_suspended": "+ukr-susp", "twibot20": "+twibot", "cp_hk": "+cp-hk"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
})


def load():
    rows = []
    with open(DATA, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["auc"] == "":
                continue
            rows.append(dict(order=r["order"], rung=int(r["rung"]), graph=r["test_graph"],
                             auc=float(r["auc"]), entry=int(r["entry_rung"]),
                             added=r["added"]))
    return rows


def role_series(rows, order):
    """Per addition step of one order: newcomer Delta, in-/out-of-mix per-graph Deltas."""
    auc = {(r["rung"], r["graph"]): r["auc"] for r in rows if r["order"] == order}
    entry = {r["graph"]: r["entry"] for r in rows if r["order"] == order}
    added = {r["rung"]: LABEL[r["added"]] for r in rows if r["order"] == order}
    newc, inc, hel = [], [], []
    for r in STEPS:
        d = {g: auc[(r, g)] - auc[(r - 1, g)] for g in entry}
        newc.append(next(d[g] for g in entry if entry[g] == r))
        inc.append([d[g] for g in entry if entry[g] < r])
        hel.append([d[g] for g in entry if entry[g] > r])
    return newc, inc, hel, [added[r] for r in STEPS]


def chrome(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def save(fig, stem):
    for ext in ("pdf", "png"):
        out = os.path.join(FIGS, f"{stem}.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print("wrote", out)
    plt.close(fig)


def subtitle(ax, msg):
    ax.text(0.0, 1.015, "NM 3-shot / 30-way · matched step 40k · within-balanced · "
            "Δ regret gap between consecutive rungs = Δ AUC (best-per-graph reference "
            "cancels)\n" + msg,
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.4, color=MUTED)


LEGEND = ["newcomer — the graph just added",
          "in-mix graphs — mean (already in the merge)",
          "out-of-mix graphs — mean (not yet in the merge)"]


def draw_grouped(ax, boost, inc_m, hel_m, ylim, errs=None, legend_loc="upper left"):
    """The shared three-bars-per-step body. errs = (n_err, i_err, h_err) or None."""
    w = 0.27
    xN = [i - w for i in X]
    xO = [i + w for i in X]
    ax.axhline(0.0, color="#9c9a93", lw=1.0, zorder=2)

    ax.bar(xN, boost, width=w, color=GREEN, edgecolor="white", linewidth=0.7, zorder=3,
           label=LEGEND[0])
    ax.bar(X, [m if m is not None else 0.0 for m in inc_m], width=w, color=CORAL,
           edgecolor="white", linewidth=0.7, zorder=3, label=LEGEND[1])
    xO2 = [x for x, m in zip(xO, hel_m) if m is not None]
    ax.bar(xO2, [m for m in hel_m if m is not None], width=w, color=GRAY,
           edgecolor="white", linewidth=0.7, zorder=3, label=LEGEND[2])

    ann_y = list(boost)
    if errs is not None:
        n_err, i_err, h_err = errs
        ax.errorbar(xN, boost, yerr=n_err, fmt="none", ecolor=GREEN_DK, elinewidth=1.1,
                    capsize=2.6, zorder=5)
        ax.errorbar(X, [m or 0.0 for m in inc_m], yerr=i_err, fmt="none", ecolor=CORAL_DK,
                    elinewidth=1.0, capsize=2.4, zorder=5)
        h_err2 = [[e[k] for e, m in zip(zip(*h_err), hel_m) if m is not None]
                  for k in (0, 1)]
        ax.errorbar(xO2, [m for m in hel_m if m is not None], yerr=h_err2, fmt="none",
                    ecolor=GRAY_DK, elinewidth=1.0, capsize=2.4, zorder=5)
        ann_y = [b + e for b, e in zip(boost, n_err[1])]

    for x, c, y in zip(xN, boost, ann_y):
        ax.annotate(f"+{c:.3f}", xy=(x, y), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.4, color=GREEN_DK,
                    fontweight="bold")

    ax.set_ylim(*ylim)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax.set_ylabel("Δ regret gap   (AUC pts; + = toward best)", fontsize=10, color=INK)
    ax.set_xlim(-0.6, len(X) - 0.4)
    ax.set_xticks(X)
    chrome(ax)
    ax.legend(loc=legend_loc, frameon=False, fontsize=9, borderaxespad=0.4,
              handlelength=1.2, labelspacing=0.45)


def mean(vals):
    return float(np.mean(vals)) if vals else None


def fig_order(order, newc, inc, hel, labels, ylim):
    inc_m = [mean(v) for v in inc]
    hel_m = [mean(v) for v in hel]

    fig, ax = plt.subplots(figsize=(10.6, 5.7), dpi=200, constrained_layout=True)
    draw_grouped(ax, newc, inc_m, hel_m, ylim,
                 legend_loc="upper right" if order == "C" else "upper left")
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_xlabel("source added to the SSL pre-training merge", fontsize=10.5, color=INK)

    im = max(abs(m) for m in inc_m if m is not None)
    hm = [m for m in hel_m if m is not None]
    if order == "C":
        title = "Order C (weakest donors first): biggest newcomer boosts — and held-out graphs recover too"
        msg = (f"every bar = Δ regret gap at that step on one shared scale · in-mix means "
               f"stay within ±{im:.3f} of zero · out-of-mix graphs GAIN "
               f"{min(hm):+.3f}…{max(hm):+.3f} per step (headroom recovery)")
    else:
        title = (f"Order {order} ({ORDER_DESC[order]}): big boost on the newcomer, "
                 "~zero impact on everyone else")
        band = max([abs(m) for m in hel_m if m is not None] + [im])
        msg = (f"every bar = Δ regret gap at that step on one shared scale · newcomer = "
               f"the added graph; the in-mix / out-of-mix means stay within "
               f"±{band:.3f} of zero")
    ax.set_title(title, fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=24)
    subtitle(ax, msg)
    save(fig, f"order_regret_combined_{order}")


def fig_minmax(per_order, ylim):
    """Bar = mean over orders A/B/C at each merge size, whisker = min/max over orders."""
    def agg(triples):
        mu = [float(np.mean(t)) for t in triples]
        lo = [m - float(np.min(t)) for m, t in zip(mu, triples)]
        hi = [float(np.max(t)) - m for m, t in zip(mu, triples)]
        return mu, [lo, hi]

    newc_t = [[per_order[o][0][i] for o in ORDERS] for i in range(len(STEPS))]
    inc_t = [[mean(per_order[o][1][i]) for o in ORDERS] for i in range(len(STEPS))]
    hel_t = [[mean(per_order[o][2][i]) for o in ORDERS] for i in range(len(STEPS))]
    hel_t = [[v for v in t if v is not None] for t in hel_t]

    n_mu, n_err = agg(newc_t)
    i_mu, i_err = agg(inc_t)
    h_mu, h_err = agg([t or [0.0] for t in hel_t])
    hel_m = [m if t else None for m, t in zip(h_mu, hel_t)]

    fig, ax = plt.subplots(figsize=(10.6, 5.7), dpi=200, constrained_layout=True)
    draw_grouped(ax, n_mu, i_mu, hel_m, ylim, errs=(n_err, i_err, h_err),
                 legend_loc="upper right")
    ax.set_xticklabels([str(s) for s in STEPS], fontsize=9.5)
    ax.set_xlabel("merge size after the addition  (the source added differs by order)",
                  fontsize=10.5, color=INK)
    ax.set_title("Adding a source, whatever the order: big boost on the newcomer, "
                 "everyone else barely moves",
                 fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=24)
    subtitle(ax, "bar = mean over orders A/B/C at that merge size, whisker = min/max over "
                 "the three orders · the newcomer differs by order · the out-of-mix upside "
                 "is order C's headroom recovery")
    save(fig, "order_regret_combined_minmax")
    return n_mu, i_mu, h_mu


def main():
    os.makedirs(FIGS, exist_ok=True)
    rows = load()
    print(f"[data] {len(rows)} cells from {os.path.relpath(DATA, HERE)}")

    per_order = {o: role_series(rows, o) for o in ORDERS}

    tops = [b for o in ORDERS for b in per_order[o][0]]
    lows = ([mean(v) for o in ORDERS for v in per_order[o][1] if v]
            + [mean(v) for o in ORDERS for v in per_order[o][2] if v])
    ylim = (min(lows) - 0.022, max(tops) * 1.14)   # one scale for all four figures

    for o in ORDERS:
        newc, inc, hel, labels = per_order[o]
        fig_order(o, newc, inc, hel, labels, ylim)
    n_mu, i_mu, h_mu = fig_minmax(per_order, ylim)

    for o in ORDERS:
        newc, inc, hel, labels = per_order[o]
        print(f"\norder {o} ({ORDER_DESC[o]})")
        print("  step        newcomer   in-mix(mean)  out-of-mix(mean)")
        for lab, b, iv, hv in zip(labels, newc, inc, hel):
            im = f"{mean(iv):+.4f}" if iv else "   —   "
            hm = f"{mean(hv):+.4f}" if hv else "— none —"
            print(f"  {lab:<10} {b:+.4f}     {im}       {hm}")
    print("\nover orders (bar heights of the min-max figure)")
    print("  size  newcomer   in-mix   out-of-mix")
    for s, n, i, h in zip(STEPS, n_mu, i_mu, h_mu):
        hm = f"{h:+.4f}" if s < 8 else "— none —"
        print(f"  {s}     {n:+.4f}   {i:+.4f}   {hm}")


if __name__ == "__main__":
    main()
