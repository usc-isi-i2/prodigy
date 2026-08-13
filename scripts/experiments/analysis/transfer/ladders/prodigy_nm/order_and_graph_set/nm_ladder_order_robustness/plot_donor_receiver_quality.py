#!/usr/bin/env python3
"""Donor / receiver quality from the order-robustness ladder, vs the specialist matrix.

The single-source matrix defines pairwise qualities (user's notebook scatter):
  specialist donor quality    = row mean over the other 7 graphs (how well models
                                trained on me serve everyone else)  [= source_outflow]
  specialist receiver quality = column mean over the other 7 models (how well models
                                trained on others serve me)         [= target_inflow]
Both are computed OFF-diagonal here (the notebook version kept the self cell).

The ladder experiment yields marginal analogs, from 3 orders:
  ladder receiver quality = the graph's mean AUC over every (order, rung) cell where
                            it is NOT yet in the merge (7-14 cells) -- how well
                            merges of other sources serve it.
  ladder marginal donor   = mean Delta AUC on the still-held-out targets at the
                            steps where the source is added (3-12 target-events).

Three panels:
  1 the ladder-derived donor x receiver plane (the new-experiment version of the
    notebook scatter);
  2 receiver quality, ladder vs specialist -- replicates (high r): being easy to
    serve is a stable property of the target graph;
  3 donor quality, ladder vs specialist -- does NOT replicate (r < 0 raw): the
    marginal "donation" is the receivers' remaining headroom, which is largest
    exactly when weak donors went first (order C). After removing the headroom
    trend (Delta ~ AUC just before the step) the donor signal is ~gone; the
    specialist matrix, not the ladder, is the instrument for donor quality.

Reads data/nm_ladder_order_robustness_long.csv and
../nm_single_source_matrix/data/nm_single_source_matrix.csv. Writes figures/.
  /opt/homebrew/bin/python3.11 plot_donor_receiver_quality.py
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
SPEC = os.path.join(HERE, "..", "nm_single_source_matrix", "data",
                    "nm_single_source_matrix.csv")
FIGS = os.path.join(HERE, "figures")

BLUE = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
ORDERS = ("A", "B", "C")

GRAPHS = ["ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
          "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter"]
GLAB = ["ukr", "covid", "midterm", "cov-pol", "elec '20", "ukr-susp", "twibot", "cp-hk"]
ADD2GRAPH = {"ukr_rus": "ukr_rus_twitter", "covid": "covid19_twitter",
             "midterm": "midterm", "covid_political": "covid_political",
             "election2020": "election2020", "ukr_rus_suspended": "ukr_rus_suspended",
             "twibot20": "twibot20", "cp_hk": "cp_hk_twitter"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
})


def load_ladder():
    rows = []
    with open(DATA, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["auc"] == "":
                continue
            rows.append(dict(order=r["order"], rung=int(r["rung"]), graph=r["test_graph"],
                             auc=float(r["auc"]), entry=int(r["entry_rung"]),
                             added=r["added"]))
    return rows


def ladder_quality(rows):
    """receiver = mean AUC while held out; donor = held-out target-event Deltas."""
    auc = {(r["order"], r["rung"], r["graph"]): r["auc"] for r in rows}
    entry = {(r["order"], r["graph"]): r["entry"] for r in rows}
    added_at = {(r["order"], r["rung"]): ADD2GRAPH[r["added"]] for r in rows}

    recv_cells = defaultdict(list)
    for r in rows:
        if r["rung"] < entry[(r["order"], r["graph"])]:
            recv_cells[r["graph"]].append(r["auc"])

    donor_events = defaultdict(list)      # src -> [(delta, before)] per held-out target
    for order in ORDERS:
        for rr in range(2, 9):
            src = added_at[(order, rr)]
            for g in GRAPHS:
                if entry[(order, g)] > rr:
                    before = auc[(order, rr - 1, g)]
                    donor_events[src].append((auc[(order, rr, g)] - before, before))

    recv = np.array([float(np.mean(recv_cells[g])) for g in GRAPHS])
    recv_n = [len(recv_cells[g]) for g in GRAPHS]
    donor = np.array([float(np.mean([d for d, _ in donor_events[g]])) for g in GRAPHS])
    donor_n = [len(donor_events[g]) for g in GRAPHS]
    return recv, recv_n, donor, donor_n, donor_events


def spec_quality():
    with open(SPEC, newline="") as fh:
        rd = list(csv.DictReader(fh))
    m = np.array([[float(row[g]) for g in GRAPHS]
                  for row in sorted(rd, key=lambda r: GRAPHS.index(r["train_graph"]))])
    off = ~np.eye(8, dtype=bool)
    donor = np.array([m[i][off[i]].mean() for i in range(8)])       # row mean, no self
    recv = np.array([m[:, j][off[:, j]].mean() for j in range(8)])  # col mean, no self
    return donor, recv


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    rk = lambda v: np.argsort(np.argsort(v))
    return float(np.corrcoef(rk(a), rk(b))[0, 1])


def chrome(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def scatter_labeled(ax, x, y, offs=None):
    """offs = {label: (dx pts, dy pts[, va])} to untangle tight label pairs."""
    ax.scatter(x, y, s=52, color=BLUE, alpha=0.85, edgecolor="white",
               linewidth=0.8, zorder=5)
    for xi, yi, lab in zip(x, y, GLAB):
        dx, dy, va = 0, 5, "bottom"
        if offs and lab in offs:
            dx, dy, *rest = offs[lab]
            va = rest[0] if rest else va
        ax.annotate(lab, (xi, yi), xytext=(dx, dy), textcoords="offset points",
                    ha="center", va=va, fontsize=8, color=INK)


def fitline(ax, x, y):
    b = np.polyfit(x, y, 1)
    xs = np.array([min(x), max(x)])
    pad = 0.06 * (xs[1] - xs[0])
    xs = np.array([xs[0] - pad, xs[1] + pad])
    ax.plot(xs, np.polyval(b, xs), color=MUTED, lw=1.3, ls=(0, (5, 3)), zorder=3)


def main():
    os.makedirs(FIGS, exist_ok=True)
    rows = load_ladder()
    l_recv, recv_n, l_donor, donor_n, donor_events = ladder_quality(rows)
    s_donor, s_recv = spec_quality()

    # headroom adjustment: one trend over all held-out target-events, mean residual/donor
    all_ev = [(d, b) for g in GRAPHS for d, b in donor_events[g]]
    dd = np.array([d for d, _ in all_ev])
    bb = np.array([b for _, b in all_ev])
    coef = np.polyfit(bb, dd, 1)
    donor_resid = np.array([
        float(np.mean([d - np.polyval(coef, b) for d, b in donor_events[g]]))
        for g in GRAPHS])

    r_recv = pearson(s_recv, l_recv)
    r_donor = pearson(s_donor, l_donor)
    r_donor_adj = pearson(s_donor, donor_resid)

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.5), dpi=200)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.755, bottom=0.115, wspace=0.28)

    # panel 1 -- the ladder-derived plane
    ax = axes[0]
    ax.axvline(0, color="#c3c2b7", lw=0.9, zorder=2)
    scatter_labeled(ax, l_donor, l_recv)
    ax.set_xlabel("ladder marginal donor effect\n(mean Δ AUC on held-out targets when added)",
                  fontsize=9.3, color=INK)
    ax.set_ylabel("ladder receiver quality\n(mean AUC while held out)", fontsize=9.3,
                  color=INK)
    chrome(ax)
    ax.set_title("the ladder's donor × receiver plane", fontsize=10.5, color=INK,
                 loc="left", pad=8)

    # panel 2 -- receiver: ladder vs specialist
    ax = axes[1]
    fitline(ax, s_recv, l_recv)
    scatter_labeled(ax, s_recv, l_recv,
                    offs={"elec '20": (-24, 3), "midterm": (12, -13, "top"),
                          "cov-pol": (10, 6)})
    ax.text(0.04, 0.90, f"r = {r_recv:+.2f}", transform=ax.transAxes, fontsize=13,
            color=INK, fontweight="bold")
    ax.set_xlabel("specialist receiver quality\n(single-source matrix, off-diag column mean)",
                  fontsize=9.3, color=INK)
    ax.set_ylabel("ladder receiver quality", fontsize=9.3, color=INK)
    chrome(ax)
    ax.set_title("receiver quality REPLICATES across designs", fontsize=10.5,
                 color=INK, loc="left", pad=8)

    # panel 3 -- donor: ladder vs specialist
    ax = axes[2]
    fitline(ax, s_donor, l_donor)
    ax.axhline(0, color="#c3c2b7", lw=0.9, zorder=2)
    scatter_labeled(ax, s_donor, l_donor, offs={"ukr": (2, -13, "top")})
    ax.text(0.04, 0.90, f"r = {r_donor:+.2f}", transform=ax.transAxes, fontsize=13,
            color=INK, fontweight="bold")
    ax.text(0.04, 0.83, f"after headroom adjustment: r = {r_donor_adj:+.2f}",
            transform=ax.transAxes, fontsize=8.6, color=MUTED)
    ax.set_xlabel("specialist donor quality\n(single-source matrix, off-diag row mean)",
                  fontsize=9.3, color=INK)
    ax.set_ylabel("ladder marginal donor effect", fontsize=9.3, color=INK)
    chrome(ax)
    ax.set_title("donor quality does NOT — it is receiver headroom", fontsize=10.5,
                 color=INK, loc="left", pad=8)

    fig.text(0.055, 0.965, "Donor & receiver quality: what the ladder can and cannot "
             "measure", fontsize=13, color=INK, fontweight="bold", ha="left", va="top")
    fig.text(0.055, 0.915,
             "NM 3-shot / 30-way · ladder receiver = mean AUC over the 7–14 cells where "
             "the graph is not yet in the merge · ladder donor = mean Δ on held-out "
             "targets over the source's 3–12 held-out (target, step) events\nspecialist "
             "qualities = off-diagonal row / column means of the single-source matrix · "
             "headroom adjustment = residual of Δ on the target's AUC just before the "
             "step (the r = −0.40 recovery trend) · inverted raw donor r: weak donors "
             "were added when headroom was largest (order C)",
             fontsize=8.3, color=MUTED, ha="left", va="top", linespacing=1.5)

    for ext in ("pdf", "png"):
        out = os.path.join(FIGS, f"order_donor_receiver_quality.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print("wrote", out)
    plt.close(fig)

    print(f"\nheadroom trend over all {len(all_ev)} held-out target-events: "
          f"Δ = {coef[0]:+.3f}·before {coef[1]:+.3f},  "
          f"corr(Δ, before) = {pearson(bb, dd):+.2f}")
    print("\ngraph       spec_donor  ladder_donor(n)   resid    spec_recv  ladder_recv(n)")
    for k in range(8):
        print(f"  {GLAB[k]:<10} {s_donor[k]:.4f}   {l_donor[k]:+.4f} ({donor_n[k]:>2})   "
              f"{donor_resid[k]:+.4f}   {s_recv[k]:.4f}    {l_recv[k]:.4f} ({recv_n[k]:>2})")
    print(f"\nreceiver: ladder vs specialist  pearson {r_recv:+.3f}  "
          f"spearman {spearman(s_recv, l_recv):+.3f}")
    print(f"donor raw: ladder vs specialist  pearson {r_donor:+.3f}  "
          f"spearman {spearman(s_donor, l_donor):+.3f}")
    print(f"donor adj: ladder vs specialist  pearson {r_donor_adj:+.3f}  "
          f"spearman {spearman(s_donor, donor_resid):+.3f}")


if __name__ == "__main__":
    main()
