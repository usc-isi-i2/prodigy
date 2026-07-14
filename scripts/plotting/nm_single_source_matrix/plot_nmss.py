#!/usr/bin/env python3
"""Plots for the 8x8 single-source NM transfer matrix.

Reads scripts/experiments/nm_single_source_matrix/nm_single_source_matrix.csv
(train_graph rows x 8 test-graph columns, NM 30-way/3-shot ROC-AUC @ matched-40k)
and writes two figures next to this script (PDF + PNG):

  1. nmss_lineplot        — x = test graph, one LINE per trained model (8 colours),
                            each model's in-domain point ringed. "Performance profile
                            of each specialist across all test graphs."
  2. nmss_rank_boxplot    — x = trained model, box = distribution of that model's RANK
                            across the 8 test-graph columns (rank 1 = best of 8 models
                            on that column). Tight-near-top = consistent generalist;
                            wide = pure specialist.

    python plot_nmss.py --csv ../../experiments/nm_single_source_matrix/nm_single_source_matrix.csv

Needs matplotlib + numpy (both in the prodigy env).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Canonical graph order + short labels (matches the ladder table / CSV columns).
GRAPHS = [
    "ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
    "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter",
]
SHORT = {
    "ukr_rus_twitter": "ukr", "covid19_twitter": "covid", "midterm": "midterm",
    "covid_political": "cov_pol", "election2020": "elec20",
    "ukr_rus_suspended": "ukr_susp", "twibot20": "twibot20", "cp_hk_twitter": "cp_hk",
}
# Okabe-Ito colourblind-safe categorical palette (8 hues, fixed order per model).
OKABE_ITO = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]  # secondary encoding (8 series)
COLOR = {g: OKABE_ITO[i] for i, g in enumerate(GRAPHS)}
MARK = {g: MARKERS[i] for i, g in enumerate(GRAPHS)}

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.dpi": 130,
})


def load_matrix(csv_path: Path) -> dict[tuple[str, str], float]:
    cells: dict[tuple[str, str], float] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            tr = row["train_graph"]
            for te in GRAPHS:
                if row.get(te):
                    cells[(tr, te)] = float(row[te])
    return cells


def plot_lineplot(cells: dict, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    x = np.arange(len(GRAPHS))
    for g in GRAPHS:  # one line per TRAINED model, fixed colour/marker
        y = [cells[(g, te)] for te in GRAPHS]
        ax.plot(x, y, color=COLOR[g], marker=MARK[g], markersize=6.5,
                linewidth=1.8, label=SHORT[g], zorder=2,
                markeredgecolor="white", markeredgewidth=0.5)
        # ring the in-domain point (test graph == train graph)
        j = GRAPHS.index(g)
        ax.plot(j, cells[(g, g)], marker=MARK[g], markersize=12, color=COLOR[g],
                markeredgecolor="black", markeredgewidth=1.4, zorder=3)
    ax.axhline(0.5, color="0.6", lw=0.8, ls=":", zorder=1)  # AUC chance
    ax.text(len(GRAPHS) - 1, 0.505, "chance (0.5)", color="0.5", fontsize=8,
            ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[g] for g in GRAPHS], rotation=20, ha="right")
    ax.set_xlabel("test graph")
    ax.set_ylabel("NM ROC-AUC (30-way / 3-shot)")
    ax.set_ylim(0.48, 1.0)
    ax.set_title("Single-source NM: each specialist's AUC profile across all test graphs\n"
                 "(ringed marker = in-domain; @matched-40k, 1 seed)", fontsize=11)
    leg = ax.legend(title="trained on", bbox_to_anchor=(1.01, 1.0), loc="upper left",
                    frameon=False, fontsize=9.5)
    leg.get_title().set_fontsize(9.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"nmss_lineplot.{ext}", bbox_inches="tight")
    plt.close(fig)


def _legend_handles():
    """One shared color/marker key (canonical order) so all figures read together."""
    return [Line2D([0], [0], linestyle="none", marker=MARK[g], markerfacecolor=COLOR[g],
                   markeredgecolor=COLOR[g], markersize=8, label=SHORT[g]) for g in GRAPHS]


def plot_boxplot(cells: dict, outdir: Path, mode: str) -> None:
    """mode='rank': per-column rank (1=best of 8). mode='delta': AUC gap to the
    best model on that column (0=best). Both: smaller = better; models ordered by
    median (best generalist left); y inverted so 'best' is at the top."""
    # per-column best AUC (for delta)
    best = {te: max(cells[(g, te)] for g in GRAPHS) for te in GRAPHS}
    vals: dict[str, list[float]] = {g: [] for g in GRAPHS}
    for te in GRAPHS:
        col = np.array([cells[(g, te)] for g in GRAPHS])
        if mode == "rank":
            order = np.argsort(-col)                       # descending AUC
            r = np.empty(len(GRAPHS), dtype=int); r[order] = np.arange(1, len(GRAPHS) + 1)
            for i, g in enumerate(GRAPHS):
                vals[g].append(int(r[i]))
        else:                                              # delta
            for g in GRAPHS:
                vals[g].append(best[te] - cells[(g, te)])
    order_models = sorted(GRAPHS, key=lambda g: (np.median(vals[g]), np.mean(vals[g])))

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    rng = np.random.default_rng(0)
    for pos, g in enumerate(order_models):
        data = vals[g]
        bp = ax.boxplot([data], positions=[pos], widths=0.6, patch_artist=True,
                        medianprops=dict(color="black", lw=1.6),
                        flierprops=dict(marker=""), zorder=2)
        for box in bp["boxes"]:
            box.set(facecolor=COLOR[g], alpha=0.30, edgecolor=COLOR[g], linewidth=1.4)
        for w in bp["whiskers"] + bp["caps"]:
            w.set(color=COLOR[g], linewidth=1.2)
        jitter = rng.uniform(-0.16, 0.16, size=len(data))
        ax.scatter(pos + jitter, data, color=COLOR[g], marker=MARK[g], s=42,
                   edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(range(len(order_models)))
    ax.set_xticklabels([SHORT[g] for g in order_models], rotation=20, ha="right")
    ax.set_xlabel("trained model")
    ax.grid(axis="x", visible=False)
    if mode == "rank":
        ax.set_ylabel("rank on a test graph  (1 = best of 8 models)")
        ax.set_yticks(range(1, len(GRAPHS) + 1))
        ax.set_ylim(len(GRAPHS) + 0.5, 0.5)               # invert: rank 1 at top
        ax.set_title("Rank consistency of each model across the 8 test graphs\n"
                     "(tight & high = universal donor; wide = pure specialist)", fontsize=11)
        fname = "nmss_rank_boxplot"
    else:
        top = max(max(v) for v in vals.values())
        ax.set_ylabel("AUC gap to the best model on that graph  (0 = best)")
        ax.set_ylim(top * 1.05, -top * 0.03)              # invert: 0 (best) at top
        ax.set_title("How far each model trails the best on each test graph\n"
                     "(tight & high = always near the top; wide = only wins in-domain)", fontsize=11)
        fname = "nmss_delta_boxplot"
    ax.legend(handles=_legend_handles(), title="model", bbox_to_anchor=(1.01, 1.0),
              loc="upper left", frameon=False, fontsize=9, handletextpad=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    ap.add_argument("--csv", default=str(
        here / "../../experiments/nm_single_source_matrix/nm_single_source_matrix.csv"))
    ap.add_argument("--outdir", default=str(here))
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cells = load_matrix(Path(args.csv))
    n = len({k[0] for k in cells})
    print(f"loaded {len(cells)} cells over {n} train sources")
    plot_lineplot(cells, outdir)
    plot_boxplot(cells, outdir, mode="rank")
    plot_boxplot(cells, outdir, mode="delta")
    print(f"wrote nmss_lineplot / nmss_rank_boxplot / nmss_delta_boxplot .(pdf|png) to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
