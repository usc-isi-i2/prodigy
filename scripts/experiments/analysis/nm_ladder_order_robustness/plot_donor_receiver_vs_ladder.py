#!/usr/bin/env python3
"""Donor / receiver quality: single-source matrix supplies it, the ladder validates it.

The order-robustness ladder alone cannot attribute DONOR quality: the mixes are
nested chains, so donor identity is confounded with merge size and order, and the
delta analysis already showed held-out gains track the receiver's headroom, not
the donor added. What the ladder does measure cleanly is the RECEIVER side (how
well a graph is served while still out of the merge, and the entry jump = what
was still missing). The single-source matrix (nm_single_source_matrix) is the
fully-crossed donor x receiver measurement. Combining them tests whether single-
donor quality COMPOSES in merges:

  panel 1  donor vs receiver quality from the matrix alone:
           x = mean AUC a source's specialist gives the other 7 (outflow),
           y = mean AUC the other 7 specialists give it (inflow).
  panel 2  merged coverage of a held-out graph vs its best in-mix donor:
           every out-of-mix cell at merge size >= 2 (63 cells over 3 orders),
           x = max over in-mix sources of SPEC[donor][target], y = merged AUC.
           On the diagonal -> a merge serves an outside graph exactly as well as
           its single best member would alone.
  panel 3  the entry-jump prediction the B/C orders were designed around:
           predicted jump = own specialist ceiling - best donor already in the
           mix (both from the matrix) vs the observed jump at entry (21 events).

Rung-1 cells are excluded from panel 2: they ARE single-source rows (B1/C1 even
reuse matrix rows verbatim), so including them would fake agreement.

Reads data/nm_ladder_order_robustness_long.csv and
../nm_single_source_matrix/data/nm_single_source_matrix.csv. Writes figures/.
  /opt/homebrew/bin/python3.11 plot_donor_receiver_vs_ladder.py
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "nm_ladder_order_robustness_long.csv")
SS = os.path.join(HERE, "..", "nm_single_source_matrix", "data",
                  "nm_single_source_matrix.csv")
FIGS = os.path.join(HERE, "figures")

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORDER_C = {"A": "#2a78d6", "B": "#e0a51f", "C": "#8f3fbf"}

GRAPHS = ["ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
          "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter"]
GLAB = ["ukr", "covid", "midterm", "cov-pol", "elec '20", "ukr-susp", "twibot", "cp-hk"]
LAB = dict(zip(GRAPHS, GLAB))

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
                             auc=float(r["auc"]), entry=int(r["entry_rung"])))
    return rows


def load_spec():
    spec = {}
    with open(SS, newline="") as fh:
        for r in csv.DictReader(fh):
            for g in GRAPHS:
                spec[(r["train_graph"], g)] = float(r[g])
    return spec


def chrome(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.grid(color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float(np.corrcoef(x, y)[0, 1])


def main():
    os.makedirs(FIGS, exist_ok=True)
    rows = load_ladder()
    spec = load_spec()
    auc = {(r["order"], r["rung"], r["graph"]): r["auc"] for r in rows}
    entry = {(r["order"], r["graph"]): r["entry"] for r in rows}

    # ---------------- panel 1 data: outflow / inflow per graph (matrix only)
    outflow = {g: np.mean([spec[(g, t)] for t in GRAPHS if t != g]) for g in GRAPHS}
    inflow = {g: np.mean([spec[(s, g)] for s in GRAPHS if s != g]) for g in GRAPHS}

    # ---------------- panel 2 data: held-out cells (merge size >= 2)
    cells = []          # (order, best in-mix donor->T, merged AUC on T)
    for order in ("A", "B", "C"):
        for rung in range(2, 8):
            mix = [g for g in GRAPHS if entry[(order, g)] <= rung]
            for t in GRAPHS:
                if entry[(order, t)] > rung:
                    best = max(spec[(d, t)] for d in mix)
                    meanp = np.mean([spec[(d, t)] for d in mix])
                    cells.append(dict(order=order, best=best, mean=float(meanp),
                                      obs=auc[(order, rung, t)], t=t))

    # ---------------- panel 3 data: entry jumps vs matrix prediction
    jumps = []
    for order in ("A", "B", "C"):
        for rung in range(2, 9):
            t = next(g for g in GRAPHS if entry[(order, g)] == rung)
            pre = [g for g in GRAPHS if entry[(order, g)] < rung]
            pred = spec[(t, t)] - max(spec[(d, t)] for d in pre)
            obs = auc[(order, rung, t)] - auc[(order, rung - 1, t)]
            jumps.append(dict(order=order, pred=pred, obs=obs, t=t))

    # ------------------------------------------------------------------ figure
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.6), dpi=200)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.76, bottom=0.125, wspace=0.27)

    # panel 1 — donor vs receiver quality
    ax = axes[0]
    xs = [outflow[g] for g in GRAPHS]
    ys = [inflow[g] for g in GRAPHS]
    ax.axvline(np.mean(xs), color=GRID, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.axhline(np.mean(ys), color=GRID, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.scatter(xs, ys, s=64, color=BLUE, edgecolor="white", linewidth=0.8, zorder=5)
    off = {"ukr": (-6, 7), "covid": (6, 5), "twibot": (6, -3), "midterm": (7, 2),
           "ukr-susp": (7, -1), "cp-hk": (7, 0), "cov-pol": (7, 0), "elec '20": (7, -4)}
    for g in GRAPHS:
        o = off.get(LAB[g], (6, 4))
        ax.annotate(LAB[g], xy=(outflow[g], inflow[g]), xytext=o,
                    textcoords="offset points", fontsize=8.6, color=INK,
                    ha="left" if o[0] > 0 else "right", va="center")
    ax.set_xlabel("donor quality — mean AUC its specialist gives the other 7",
                  fontsize=9.3, color=INK)
    ax.set_ylabel("receiver quality — mean AUC the other 7 give it",
                  fontsize=9.3, color=INK)
    chrome(ax)
    ax.set_title("who gives well, who receives well\n(single-source matrix alone)",
                 fontsize=10.2, color=INK, loc="left", pad=8)

    # panel 2 — merged coverage vs best in-mix donor
    ax = axes[1]
    lims = (0.52, 1.0)
    ax.plot(lims, lims, color=MUTED, lw=1.2, ls=(0, (5, 3)), zorder=2)
    for order in ("A", "B", "C"):
        pts = [c for c in cells if c["order"] == order]
        ax.scatter([c["best"] for c in pts], [c["obs"] for c in pts], s=40,
                   color=ORDER_C[order], alpha=0.75, edgecolor="white", linewidth=0.5,
                   zorder=5, label=f"order {order}")
    resid = [c["obs"] - c["best"] for c in cells]
    r_best = pearson([c["best"] for c in cells], [c["obs"] for c in cells])
    r_mean = pearson([c["mean"] for c in cells], [c["obs"] for c in cells])
    ax.text(0.04, 0.93, f"r = {r_best:+.2f}   (mean-donor r = {r_mean:+.2f})",
            transform=ax.transAxes, fontsize=9.5, color=INK, fontweight="bold")
    ax.text(0.04, 0.865, f"merge − best donor: mean {np.mean(resid):+.3f}, "
            f"{sum(v > 0 for v in resid)}/{len(resid)} above",
            transform=ax.transAxes, fontsize=8.4, color=MUTED)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("best single-source transfer to it from any in-mix source",
                  fontsize=9.3, color=INK)
    ax.set_ylabel("merged model's AUC on the held-out graph", fontsize=9.3, color=INK)
    chrome(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=8.6, handlelength=1.0)
    ax.set_title("a merge serves an outside graph like\nits best member would alone",
                 fontsize=10.2, color=INK, loc="left", pad=8)

    # panel 3 — entry jump vs matrix prediction
    ax = axes[2]
    lims3 = (-0.02, 0.40)
    ax.plot(lims3, lims3, color=MUTED, lw=1.2, ls=(0, (5, 3)), zorder=2)
    for order in ("A", "B", "C"):
        pts = [j for j in jumps if j["order"] == order]
        ax.scatter([j["pred"] for j in pts], [j["obs"] for j in pts], s=44,
                   color=ORDER_C[order], alpha=0.8, edgecolor="white", linewidth=0.5,
                   zorder=5, label=f"order {order}")
    r_j = pearson([j["pred"] for j in jumps], [j["obs"] for j in jumps])
    mae = np.mean([abs(j["obs"] - j["pred"]) for j in jumps])
    bias = np.mean([j["obs"] - j["pred"] for j in jumps])
    ax.text(0.04, 0.93, f"r = {r_j:+.2f}", transform=ax.transAxes, fontsize=11,
            color=INK, fontweight="bold")
    ax.text(0.04, 0.865, f"MAE {mae:.3f} · obs − pred mean {bias:+.3f}",
            transform=ax.transAxes, fontsize=8.4, color=MUTED)
    ax.set_xlim(*lims3)
    ax.set_ylim(*lims3)
    ax.set_xlabel("predicted jump = own ceiling − best donor already in the mix\n"
                  "(both from the single-source matrix)", fontsize=9.3, color=INK)
    ax.set_ylabel("observed jump at entry into the merge", fontsize=9.3, color=INK)
    chrome(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=8.6, handlelength=1.0)
    ax.set_title("the entry jump is the unserved part:\nceiling − best donor present",
                 fontsize=10.2, color=INK, loc="left", pad=8)

    fig.text(0.055, 0.965, "Donor and receiver quality: measured by the single-source "
             "matrix, validated by the ladder", fontsize=13, color=INK,
             fontweight="bold", ha="left", va="top")
    fig.text(0.055, 0.915,
             "NM 3-shot / 30-way · matched step 40k · panel 1 = matrix row/column "
             "means (specialist models only) · panels 2–3 = the 3-order ladder scored "
             "against matrix-derived predictors\npanel 2: every held-out (graph, rung) "
             "cell at merge size ≥ 2 (63 cells; rung-1 cells excluded — they are the "
             "single-source rows themselves) · panel 3: all 21 measurable entry events",
             fontsize=8.4, color=MUTED, ha="left", va="top", linespacing=1.5)

    for ext in ("pdf", "png"):
        out = os.path.join(FIGS, f"order_donor_receiver.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print("wrote", out)
    plt.close(fig)

    # ---------------------------------------------------------------- console
    print("\ngraph      donor(outflow)  receiver(inflow)  ceiling  ceiling−inflow")
    for g in GRAPHS:
        print(f"  {LAB[g]:<9} {outflow[g]:.4f}         {inflow[g]:.4f}          "
              f"{spec[(g, g)]:.4f}   {spec[(g, g)] - inflow[g]:+.4f}")
    print(f"\npanel 2 (n={len(cells)}): r(best-donor) = {r_best:+.3f}, "
          f"r(mean-donor) = {r_mean:+.3f}")
    print(f"  merge − best donor: mean {np.mean(resid):+.4f}, median "
          f"{np.median(resid):+.4f}, range {min(resid):+.4f}..{max(resid):+.4f}, "
          f"{sum(v > 0 for v in resid)}/{len(resid)} above the donor")
    print(f"\npanel 3 (n={len(jumps)}): r = {r_j:+.3f}, MAE = {mae:.4f}, "
          f"obs − pred mean {bias:+.4f}")
    worst = max(jumps, key=lambda j: abs(j["obs"] - j["pred"]))
    print(f"  worst event: {LAB[worst['t']]} order {worst['order']} "
          f"pred {worst['pred']:+.3f} obs {worst['obs']:+.3f}")


if __name__ == "__main__":
    main()
