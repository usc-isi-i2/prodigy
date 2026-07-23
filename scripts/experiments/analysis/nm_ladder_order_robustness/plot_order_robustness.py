#!/usr/bin/env python3
"""Order-robustness figures: the three single-order ladder plots, aggregated over orders.

The published NM ladder (nm_ladder_fillin) added the 8 source graphs in ONE order. This
experiment reran it under three orders (A published, B donor-strength descending, C its
reverse). These three figures are the order-aggregated counterparts of the three original
ladder figures, so the "adding a graph" story can be told without depending on a sequence:

  fig 1  entry-aligned trajectory   <- plot_nm_ladder.py (trajectory / diagonal cascade)
         Each test graph's AUC re-indexed by rungs-relative-to-its-own-entry, every
         (graph, order) overlaid, with the mean and min/max band. The absolute rung a
         graph enters differs by order; aligned on entry, the jump is one shape.

  fig 2  in-/out-of-distribution gap <- plot_nm_ladder_means.py (ID vs all-8 mean)
         Per rung (= merge size), the in-distribution mean vs the all-8 mean, averaged
         over the three orders with a min/max band. Rung k = k sources in every order, so
         this axis IS comparable across orders. The gap closes regardless of order.

  fig 3  per-source role decomposition <- plot_nm_ladder_deltas.py (newcomer/incumbent/held-out)
         Every source-addition event, pooled over the three orders (7 rungs x 3 = 21
         newcomer events), split by role: newcomer (its own OOD->ID jump = benefit),
         incumbents (already in training = cost), held-out (not yet added = interference).
         Distribution, not a single order's 7 bars.

Reads data/nm_ladder_order_robustness_long.csv (entry-aligned long form from
assemble_order_table.py). Writes figures/ here. Local homebrew python:
  /opt/homebrew/bin/python3.11 plot_order_robustness.py
"""
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "nm_ladder_order_robustness_long.csv")
FIGS = os.path.join(HERE, "figures")

# palette (matches the nm_ladder figures)
BLUE = "#2a78d6"    # in-distribution / newcomer benefit
CORAL = "#d85a30"   # incumbents (cost)
GRAY = "#8f8d87"    # held-out / out-of-distribution
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
ORDER_C = {"A": "#2a78d6", "B": "#e0a51f", "C": "#8f3fbf"}

RUNGS = list(range(1, 9))

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
            rows.append(dict(
                order=r["order"], rung=int(r["rung"]), n_sources=int(r["n_sources"]),
                graph=r["test_graph"], auc=float(r["auc"]),
                entry_rung=int(r["entry_rung"]), rel=int(r["rel_to_entry"]),
                in_training=r["in_training"] == "1",
            ))
    return rows


def chrome(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)


def save(fig, stem):
    for ext in ("pdf", "png"):
        out = os.path.join(FIGS, f"{stem}.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print("wrote", out)


# --------------------------------------------------------------------- figure 1
def fig_entry_aligned(rows):
    """Every (graph, order) AUC curve aligned on its own entry rung, + mean and band."""
    by_pair = defaultdict(dict)                       # (graph, order) -> {rel: auc}
    for r in rows:
        by_pair[(r["graph"], r["order"])][r["rel"]] = r["auc"]

    rels = list(range(-4, 8))
    at_rel = defaultdict(list)
    for series in by_pair.values():
        for rel, auc in series.items():
            at_rel[rel].append(auc)
    # keep offsets covered by a healthy number of (graph, order) pairs
    keep = [rel for rel in rels if len(at_rel.get(rel, [])) >= 6]
    mean = [float(np.mean(at_rel[rel])) for rel in keep]
    lo = [float(np.min(at_rel[rel])) for rel in keep]
    hi = [float(np.max(at_rel[rel])) for rel in keep]

    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=200)

    for (graph, order), series in by_pair.items():
        xs = sorted(series)
        ax.plot(xs, [series[x] for x in xs], color=GRAY, lw=0.7, alpha=0.28, zorder=2)

    ax.fill_between(keep, lo, hi, color=BLUE, alpha=0.10, zorder=3, linewidth=0)
    ax.plot(keep, mean, color=INK, lw=2.8, zorder=6, marker="o", ms=6,
            markerfacecolor=INK, markeredgecolor="white", markeredgewidth=1.2)

    ax.axvline(0, color=CORAL, lw=1.4, ls=(0, (3, 2)), zorder=4)
    ax.annotate("graph enters\nthe merge", xy=(0.12, 0.775), fontsize=9,
                color=CORAL, ha="left", va="center", fontweight="bold")

    # the mean jump at entry
    m = dict(zip(keep, mean))
    if -1 in m and 0 in m:
        ax.annotate(f"mean jump +{m[0]-m[-1]:.3f}",
                    xy=(0, m[0]), xytext=(1.2, m[0] - 0.055),
                    fontsize=9.5, color=INK, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))

    ax.set_xlim(min(keep) - 0.4, max(keep) + 0.4)
    ax.set_xticks(keep)
    ax.set_xlabel("rungs relative to the graph's own entry into the merge  "
                  "(0 = entry; <0 held out, >0 in training)", fontsize=10.3, color=INK)
    ax.set_ylabel("NM AUC  (3-shot, 30-way)", fontsize=10.5, color=INK)
    chrome(ax)
    ax.set_title("Entering the merge causes the jump — in every order",
                 fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=26)
    ax.text(0.0, 1.02, "each thin line = one (graph, order); bold = mean over all "
            "24 (graph, order) pairs, band = min/max  ·  21/21 entry jumps positive, "
            "sign-test p = 5e-7", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.6, color=MUTED)
    ax.legend(handles=[
        Line2D([0], [0], color=GRAY, lw=1.2, alpha=0.5, label="individual (graph, order)"),
        Line2D([0], [0], color=INK, lw=2.8, marker="o", markerfacecolor=INK,
               markeredgecolor="white", ms=7, label="mean (24 pairs)"),
    ], loc="lower right", frameon=False, fontsize=9, handlelength=2.4)

    fig.tight_layout()
    save(fig, "order_entry_aligned_trajectory")
    plt.close(fig)


# --------------------------------------------------------------------- figure 2
def fig_id_ood_gap(rows):
    """In-distribution mean vs all-8 mean per rung, averaged over the three orders."""
    # per (order, rung): all-8 mean and in-dist mean (graphs already in the merge)
    by_or = defaultdict(lambda: defaultdict(list))    # (order, rung) -> {'all','in'}
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["order"], r["rung"])].append(r)
    all_mean = defaultdict(list)   # rung -> [order means]
    in_mean = defaultdict(list)
    for (order, rung), rs in grouped.items():
        all_mean[rung].append(float(np.mean([x["auc"] for x in rs])))
        inc = [x["auc"] for x in rs if x["in_training"]]
        in_mean[rung].append(float(np.mean(inc)))

    x = [rg - 1 for rg in RUNGS]
    all_mu = [float(np.mean(all_mean[rg])) for rg in RUNGS]
    all_lo = [float(np.min(all_mean[rg])) for rg in RUNGS]
    all_hi = [float(np.max(all_mean[rg])) for rg in RUNGS]
    in_mu = [float(np.mean(in_mean[rg])) for rg in RUNGS]
    in_lo = [float(np.min(in_mean[rg])) for rg in RUNGS]
    in_hi = [float(np.max(in_mean[rg])) for rg in RUNGS]

    fig, ax = plt.subplots(figsize=(8.8, 5.3), dpi=200)
    ax.fill_between(x, in_mu, all_mu, color=BLUE, alpha=0.10, zorder=1, linewidth=0)
    ax.fill_between(x, in_lo, in_hi, color=BLUE, alpha=0.16, zorder=2, linewidth=0)
    ax.fill_between(x, all_lo, all_hi, color=INK, alpha=0.10, zorder=2, linewidth=0)
    ax.plot(x, in_mu, color=BLUE, lw=2.6, zorder=5, marker="o", ms=6.5,
            markerfacecolor=BLUE, markeredgecolor="white", markeredgewidth=1.2)
    ax.plot(x, all_mu, color=INK, lw=2.6, zorder=6, marker="s", ms=6,
            markerfacecolor=INK, markeredgecolor="white", markeredgewidth=1.2)

    ax.annotate("in training\n(in-dist. mean)", xy=(x[-1] + 0.12, in_mu[-1] + 0.004),
                ha="left", va="center", fontsize=9.5, color=BLUE, fontweight="bold")
    ax.annotate("all 8 graphs\n(incl. held-out)", xy=(x[-1] + 0.12, all_mu[-1] - 0.012),
                ha="left", va="center", fontsize=9.5, color=INK, fontweight="bold")
    gi = 1
    ax.annotate("", xy=(gi, in_mu[gi]), xytext=(gi, all_mu[gi]),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.1))
    ax.annotate("out-of-distribution\npenalty", xy=(gi + 0.12, (in_mu[gi] + all_mu[gi]) / 2),
                ha="left", va="center", fontsize=8.8, color=MUTED)

    ax.set_xlim(-0.5, 8.7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(rg) for rg in RUNGS], fontsize=9.5)
    ax.set_xlabel("merge size  (number of source graphs in SSL pre-training)",
                  fontsize=10.5, color=INK)
    ax.set_ylabel("mean NM AUC  (3-shot, 30-way)", fontsize=10.5, color=INK)
    chrome(ax)
    ax.set_title("Adding sources closes the in-/out-of-distribution gap — order-robustly",
                 fontsize=12.3, color=INK, fontweight="bold", loc="left", pad=26)
    ax.text(0.0, 1.02, "lines = mean over orders A/B/C, bands = min/max over orders  ·  "
            "in-dist. mean = graphs already in the merge at that size",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.6, color=MUTED)
    ax.legend(handles=[
        Line2D([0], [0], color=BLUE, lw=2.6, marker="o", markerfacecolor=BLUE,
               markeredgecolor="white", ms=7, label="in training (in-dist. mean)"),
        Line2D([0], [0], color=INK, lw=2.6, marker="s", markerfacecolor=INK,
               markeredgecolor="white", ms=7, label="all 8 graphs (incl. held-out)"),
    ], loc="lower right", frameon=False, fontsize=9, handlelength=2.4)

    fig.tight_layout()
    save(fig, "order_id_ood_gap")
    plt.close(fig)


# --------------------------------------------------------------------- figure 3
def fig_role_deltas(rows):
    """Per-source impact split by role, pooled over the three orders."""
    # index AUC by (order, rung, graph) and by (order, graph)->entry
    auc = {(r["order"], r["rung"], r["graph"]): r["auc"] for r in rows}
    entry = {}
    graphs_by_order = defaultdict(set)
    for r in rows:
        entry[(r["order"], r["graph"])] = r["entry_rung"]
        graphs_by_order[r["order"]].add(r["graph"])

    newc, inc, hel = [], [], []
    for order in ("A", "B", "C"):
        graphs = graphs_by_order[order]
        for rr in range(2, 9):                                  # rung rr vs rr-1
            deltas = {}
            for g in graphs:
                a0 = auc.get((order, rr - 1, g))
                a1 = auc.get((order, rr, g))
                if a0 is not None and a1 is not None:
                    deltas[g] = a1 - a0
            newcomer = next(g for g in graphs if entry[(order, g)] == rr)
            incs = [g for g in graphs if entry[(order, g)] < rr]
            hels = [g for g in graphs if entry[(order, g)] > rr]
            if newcomer in deltas:
                newc.append(deltas[newcomer])
            inc += [deltas[g] for g in incs if g in deltas]
            hel += [deltas[g] for g in hels if g in deltas]

    groups = [("newcomer\n(just added)", newc, BLUE),
              ("incumbents\n(already in)", inc, CORAL),
              ("held-out\n(not yet added)", hel, GRAY)]

    fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=200)
    rng = np.random.default_rng(0)
    for i, (lab, vals, col) in enumerate(groups):
        vals = np.array(vals)
        jitter = rng.uniform(-0.13, 0.13, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=26, color=col, alpha=0.55,
                   edgecolor="white", linewidth=0.4, zorder=4)
        mu = float(vals.mean())
        ax.plot([i - 0.28, i + 0.28], [mu, mu], color=INK, lw=2.4, zorder=6)
        ax.annotate(f"mean {mu:+.3f}\nn={len(vals)}", xy=(i + 0.33, mu),
                    fontsize=8.8, color=col if i == 0 else MUTED, va="center",
                    fontweight="bold" if i == 0 else "normal")

    ax.axhline(0, color="#c3c2b7", lw=1.0, zorder=2)
    ax.set_xticks(range(3))
    ax.set_xticklabels([g[0] for g in groups], fontsize=9.8)
    ax.set_ylabel("Δ NM AUC at each source-addition step\n(this rung − rung below)",
                  fontsize=10.3, color=INK)
    ax.set_xlim(-0.6, 2.9)
    chrome(ax)
    ax.set_title("Per-source impact: large in-domain benefit, ~zero cost to other graphs",
                 fontsize=12.0, color=INK, fontweight="bold", loc="left", pad=26)
    ax.text(0.0, 1.02, "each dot = one graph at one addition step, pooled over orders "
            "A/B/C  ·  newcomer = its own OOD-to-ID jump  ·  21 newcomer events, all positive",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8.6, color=MUTED)

    fig.tight_layout()
    save(fig, "order_role_deltas")
    plt.close(fig)

    # console summary
    print("\nrole            n     mean      min      max")
    for lab, vals, _ in groups:
        v = np.array(vals)
        print(f"  {lab.split(chr(10))[0]:<14}{len(v):<4}{v.mean():+.4f}  {v.min():+.4f}  {v.max():+.4f}")


def main():
    os.makedirs(FIGS, exist_ok=True)
    rows = load()
    print(f"[data] {len(rows)} cells from {os.path.relpath(DATA, HERE)}")
    fig_entry_aligned(rows)
    fig_id_ood_gap(rows)
    fig_role_deltas(rows)


if __name__ == "__main__":
    main()
