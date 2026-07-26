#!/usr/bin/env python3
"""Similarity vs. transfer, full 8x8 single-source matrix (confound-free upgrade of
the 4x5 pilot). Joins graph-divergence metrics with the single-source NM transfer
matrix and renders two figures (PDF + PNG):

  1. nmss_donor_vs_separation  — scatter: x = a graph's mean feature-cloud separation
     (proxy-A distance to the other 7), y = its donor strength (mean NM accuracy it
     transfers OUT). One point per graph; OLS guide + Spearman rho. The "knockout".
  2. nmss_rho_by_metric        — within-target Spearman rho (source divergence vs.
     transfer) per similarity metric, coloured by axis (feature vs topology).

    python plot_sim_transfer_8x8.py \
        --div-json ../graph_divergence/graph_divergence_data.json \
        --transfer-csv ../../experiments/nm_single_source_matrix/nm_single_source_matrix_long.csv

Needs matplotlib + numpy (prodigy env). Stats are computed here (stdlib), not hardcoded.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CANON = [
    "ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
    "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter",
]
SHORT = {
    "ukr_rus_twitter": "ukr", "covid19_twitter": "covid", "midterm": "midterm",
    "covid_political": "cov_pol", "election2020": "elec20",
    "ukr_rus_suspended": "ukr_susp", "twibot20": "twibot20", "cp_hk_twitter": "cp_hk",
}
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73",
             "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
COLOR = {g: OKABE_ITO[i] for i, g in enumerate(CANON)}
MARK = {g: m for g, m in zip(CANON, ["o", "s", "^", "D", "v", "P", "X", "*"])}

# similarity metrics: (json key, display label, axis)
METRICS = [
    ("proxy_a_distance", "proxy-A distance\n(feature separability)", "feature"),
    ("feat_frechet", "Fréchet (feature)", "feature"),
    ("feat_mmd2", "RBF-MMD² (feature)", "feature"),
    ("feat_centroid_cosdist", "centroid cosine (feature)", "feature"),
    ("indegree_ks", "in-degree KS (topology)", "topology"),
    ("outdegree_ks", "out-degree KS (topology)", "topology"),
]
AXIS_COLOR = {"feature": "#0072B2", "topology": "#D55E00"}

# label offsets for the scatter (data units) so close points don't collide
LABELPOS = {
    "ukr_rus_twitter": (-0.012, 0.005, "right", "bottom"),
    "covid19_twitter": (0.012, -0.006, "left", "top"),
    "twibot20": (0.014, 0.001, "left", "center"),
    "ukr_rus_suspended": (0.014, 0.0, "left", "center"),
    "midterm": (0.014, 0.0, "left", "center"),
    "cp_hk_twitter": (0.014, 0.0, "left", "center"),
    "covid_political": (0.0, 0.007, "center", "bottom"),
    "election2020": (0.0, -0.007, "center", "top"),
}

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6, "figure.dpi": 130,
})


def rankdata(a):
    order = sorted(range(len(a)), key=lambda i: a[i]); rk = [0.0] * len(a); i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        for k in range(i, j + 1):
            rk[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return rk


def pearson(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x); syy = sum((b - my) ** 2 for b in y)
    return float("nan") if sxx == 0 or syy == 0 else sxy / math.sqrt(sxx * syy)


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def load(div_json: Path, transfer_csv: Path):
    div = json.load(open(div_json))
    gj = div["graphs"]; jidx = {g: i for i, g in enumerate(gj)}
    pw = div["pairwise"]
    sim = {m: (lambda M: (lambda s, t: pw[M][jidx[s]][jidx[t]]))(m) for m, _, _ in METRICS}
    acc = {}
    with open(transfer_csv) as f:
        for r in csv.DictReader(f):
            if r["metric"] == "accuracy":
                acc[(r["train"], r["test"])] = float(r["value"])
    return sim, acc


def within_target_rho(sim_fn, acc, exclude_self):
    rhos = []
    for t in CANON:
        srcs = [s for s in CANON if not (exclude_self and s == t)]
        xs = [sim_fn(s, t) for s in srcs]
        ys = [acc[(s, t)] for s in srcs]
        rhos.append(spearman(xs, ys))
    good = [r for r in rhos if not math.isnan(r)]
    return sum(good) / len(good)


def _r2(ys, pred):
    ybar = ys.mean()
    return 1 - float(((ys - pred) ** 2).sum() / ((ys - ybar) ** 2).sum())


def plot_scatter(sim, acc, outdir: Path, metric_key, xlabel, title, fname, labelpos=None,
                 logx=False, logy=False):
    sep = {g: np.mean([sim[metric_key](g, o) for o in CANON if o != g]) for g in CANON}
    donor = {g: np.mean([acc[(g, o)] for o in CANON if o != g]) for g in CANON}
    xs = np.array([sep[g] for g in CANON]); ys = np.array([donor[g] for g in CANON])
    rho = spearman(list(xs), list(ys))
    xr = xs.max() - xs.min() or 1.0

    lx, ly = np.log(xs), np.log(ys)

    def loo_r2(px, log_y):                 # px transformed; log_y=True -> exp back-transform
        py = ly if log_y else ys
        pr = np.empty_like(ys)
        for i in range(len(xs)):
            j = [t for t in range(len(xs)) if t != i]
            s, c = np.polyfit(px[j], py[j], 1)
            v = c + s * px[i]
            pr[i] = math.exp(v) if log_y else v
        return _r2(ys, pr)

    m, b = np.polyfit(xs, ys, 1)                                   # linear
    r2_lin, loo_lin = _r2(ys, m * xs + b), loo_r2(xs, False)
    ke, lnAe = np.polyfit(xs, ly, 1); Ae = math.exp(lnAe)         # exponential y=Ae*exp(ke*x)
    r2_exp, loo_exp = _r2(ys, Ae * np.exp(ke * xs)), loo_r2(xs, True)
    pe, lnAp = np.polyfit(lx, ly, 1); Ap = math.exp(lnAp)         # power y=Ap*x^pe
    r2_pow, loo_pow = _r2(ys, Ap * xs ** pe), loo_r2(lx, True)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    if logx:
        ax.set_xscale("log")
        xl = np.geomspace(xs.min() * 0.92, xs.max() * 1.08, 120)
    else:
        xl = np.linspace(max(1e-3, xs.min() - 0.05 * xr), xs.max() + 0.05 * xr, 100)
    if logy:
        ax.set_yscale("log")                 # log-log -> the power-law fit is a straight line
    ax.plot(xl, m * xl + b, color="0.75", lw=1.2, ls="--", zorder=1,
            label=f"linear  (R²={r2_lin:.2f}, LOO={loo_lin:.2f})")
    ax.plot(xl, Ae * np.exp(ke * xl), color="0.5", lw=1.6, zorder=2,
            label=f"exponential  (R²={r2_exp:.2f}, LOO={loo_exp:.2f})")
    ax.plot(xl, Ap * xl ** pe, color="#111", lw=2.5, zorder=2,
            label=f"power  (R²={r2_pow:.2f}, LOO={loo_pow:.2f})")
    for g in CANON:
        ax.scatter(sep[g], donor[g], color=COLOR[g], marker=MARK[g], s=150,
                   edgecolor="white", linewidth=0.8, zorder=3)
        col = COLOR[g] if g != "election2020" else "#8a7a00"
        if logx or logy:   # additive data-unit offsets don't work on a log axis -> pixel offsets
            ax.annotate(SHORT[g], (sep[g], donor[g]), xytext=(7, 5),
                        textcoords="offset points", ha="left", va="bottom",
                        fontsize=10, color=col, fontweight="bold")
        else:
            dx, dy, ha, va = (labelpos or {}).get(g, (0.02 * xr, 0.006, "left", "bottom"))
            ax.annotate(SHORT[g], (sep[g], donor[g]), xytext=(sep[g] + dx, donor[g] + dy),
                        ha=ha, va=va, fontsize=10, color=col, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("donor strength\n(mean NM accuracy transferred to other graphs)")
    ax.set_title(f"{title}\nSpearman ρ = {rho:+.2f};  best fit  donor = {Ap:.2f}·d^({pe:+.2f}),  n = 8",
                 fontsize=11)
    ax.legend(frameon=False, loc="upper right", fontsize=9, title="fit (each 2 params)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)
    return rho


def plot_rho_bars(sim, acc, outdir: Path):
    rows = []
    for key, label, axis in METRICS:
        rows.append((label, axis,
                     within_target_rho(sim[key], acc, exclude_self=True),
                     within_target_rho(sim[key], acc, exclude_self=False)))
    rows.sort(key=lambda z: z[2])                      # most negative (strongest) first
    labels = [z[0] for z in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for i, (label, axis, rho_ns, rho_s) in enumerate(rows):
        ax.barh(i, rho_ns, color=AXIS_COLOR[axis], alpha=0.85, height=0.62, zorder=2)
        ax.plot(rho_s, i, marker="|", markersize=16, color="0.15", zorder=3)  # incl-self
        ax.text(-0.02, i, f"{rho_ns:+.2f}", va="center", ha="right",
                fontsize=9.5, color="white", fontweight="bold")
    ax.axvline(0, color="0.3", lw=0.9)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()                                  # strongest at top
    ax.set_xlabel("within-target Spearman ρ  (source divergence vs. NM transfer)\nmore negative = better predictor")
    ax.set_xlim(-0.85, 0.05)
    ax.grid(axis="y", visible=False)
    ax.set_title("Feature-cloud separability predicts transfer; topology is weaker\n"
                 "(bar = self-excluded ρ, mean over 8 targets; │ tick = with-self)", fontsize=11)
    handles = [Line2D([0], [0], marker="s", linestyle="none", markersize=10,
                      markerfacecolor=AXIS_COLOR[a], markeredgecolor=AXIS_COLOR[a],
                      label=f"{a} axis") for a in ("feature", "topology")]
    handles.append(Line2D([0], [0], marker="|", linestyle="none", markersize=12,
                          color="0.15", label="ρ incl. self"))
    ax.legend(handles=handles, frameon=False, loc="lower left", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"nmss_rho_by_metric.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--div-json", default=str(here / "../graph_divergence/graph_divergence_data.json"))
    ap.add_argument("--transfer-csv", default=str(
        here / "../../experiments/nm_single_source_matrix/nm_single_source_matrix_long.csv"))
    ap.add_argument("--outdir", default=str(here))
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    sim, acc = load(Path(args.div_json), Path(args.transfer_csv))
    rho_feat = plot_scatter(
        sim, acc, outdir, "proxy_a_distance",
        "feature-cloud separation\n(mean proxy-A distance to the other 7 graphs)",
        "A graph's feature-cloud outlier-ness predicts how well it donates",
        "nmss_donor_vs_separation", labelpos=LABELPOS)
    plot_scatter(
        sim, acc, outdir, "proxy_a_distance",
        "feature-cloud separation — log scale\n(mean proxy-A distance to the other 7 graphs)",
        "Feature distance vs. donor strength (log-log: power law = straight line)",
        "nmss_donor_vs_separation_loglog", logx=True, logy=True)
    rho_ind = plot_scatter(
        sim, acc, outdir, "indegree_ks",
        "topology distance\n(mean in-degree KS distance to the other 7 graphs)",
        "Donor strength vs. topology (in-degree) distance",
        "nmss_donor_vs_indeg_ks")
    plot_scatter(
        sim, acc, outdir, "indegree_ks",
        "topology distance — log scale\n(mean in-degree KS distance to the other 7 graphs)",
        "Donor strength vs. in-degree distance (log x)",
        "nmss_donor_vs_indeg_ks_logx", logx=True)
    plot_scatter(
        sim, acc, outdir, "indegree_ks",
        "topology distance — log scale\n(mean in-degree KS distance to the other 7 graphs)",
        "In-degree distance vs. donor strength (log-log)",
        "nmss_donor_vs_indeg_ks_loglog", logx=True, logy=True)
    rho_out = plot_scatter(
        sim, acc, outdir, "outdegree_ks",
        "topology distance\n(mean out-degree KS distance to the other 7 graphs)",
        "Out-degree distance vs. donor strength",
        "nmss_donor_vs_outdeg_ks")
    plot_rho_bars(sim, acc, outdir)
    print(f"donor-vs-distance Spearman rho:  proxy-A={rho_feat:+.3f}  "
          f"indeg_ks={rho_ind:+.3f}  outdeg_ks={rho_out:+.3f}")
    print(f"wrote scatters + nmss_rho_by_metric .(pdf|png) to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
