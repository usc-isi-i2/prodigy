#!/usr/bin/env python3
"""Slide figures for the frozen-probe cross-task matrix (pretrain_probe_matrix / Version B).

Reads the shared benchmark CSVs (source of truth) + the features_only floor, aggregates
per model (mean over datasets), and renders two slide-ready figures:

  fig1  probe_matrix_heatmap        models x {node-reg, static-LP}, cells colored by
                                    margin over the random-encoder floor.
  fig2  probe_matrix_delta_bars     diverging Delta-over-floor bars (the punchline).

No pandas (csv + numpy only). Saves PNG (dpi 400) + PDF, matching best_pretrain_strat.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REG_CSV = ROOT / "scripts/experiments/analysis/node_regression/data/node_regression.csv"
SLP_CSV = ROOT / "scripts/experiments/analysis/static_link_prediction/data/static_link_prediction.csv"
FEAT_CSV = ROOT / "scripts/experiments/analysis/node_regression/data/features_only_floor.csv"
OUT = Path(__file__).resolve().parent

# rows (top->bottom): the 4 pretraining strategies, then the two floors.
STRATS = ["task_transfer_covid_nm", "task_transfer_covid_cl",
          "task_transfer_covid_fp", "muc10k"]
LABELS = {
    "task_transfer_covid_nm": ("NM · covid", "neighbor matching"),
    "task_transfer_covid_cl": ("CL · covid", "contrastive"),
    "task_transfer_covid_fp": ("FP · covid", "masked feature pred."),
    "muc10k":                 ("NM · merged", "ukr+covid"),
    "random_init":            ("random encoder", "untrained, same arch"),
    "features_only":          ("raw features", "ridge, no GNN"),
}
FLOOR = "random_init"

# diverging: red (below floor) -> neutral -> blue (beats floor)
CMAP = LinearSegmentedColormap.from_list(
    "floor_div", ["#A32D2D", "#E24B4A", "#F1EFE8", "#378ADD", "#0C447C"])
VLIM = 0.16


def _mean(path: Path, model: str, value_col: str, shot: str) -> float:
    vals = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["model"] == model and r.get("split", "test") == "test" and r["shots"] == shot:
                try:
                    vals.append(float(r[value_col]))
                except (ValueError, KeyError):
                    pass
    return float(np.mean(vals)) if vals else np.nan


def _features_only_reg() -> float:
    if not FEAT_CSV.exists():
        return np.nan
    vals = []
    with open(FEAT_CSV) as f:
        for r in csv.DictReader(f):
            v = r.get("spearman", "")
            try:
                fv = float(v)
                if np.isfinite(fv):
                    vals.append(fv)
            except ValueError:
                pass
    return float(np.mean(vals)) if vals else np.nan


def collect():
    reg, slp = {}, {}
    for m in STRATS + [FLOOR]:
        reg[m] = _mean(REG_CSV, m, "spearman", "10")
        slp[m] = _mean(SLP_CSV, m, "roc_auc", "0")
    reg["features_only"] = _features_only_reg()
    slp["features_only"] = np.nan  # features_only is a node-target floor, not an LP floor
    return reg, slp


def main() -> int:
    reg, slp = collect()
    rows = STRATS + [FLOOR, "features_only"]
    reg_floor, slp_floor = reg[FLOOR], slp[FLOOR]

    print(f"{'model':<24}{'reg ρ':>9}{'Δreg':>8}{'slp AUC':>10}{'Δslp':>8}")
    for m in rows:
        dr = reg[m] - reg_floor if np.isfinite(reg[m]) else np.nan
        ds = slp[m] - slp_floor if np.isfinite(slp[m]) else np.nan
        print(f"{m:<24}{reg[m]:>9.3f}{dr:>8.3f}{slp[m]:>10.3f}{ds:>8.3f}")

    # ---- fig 1: heatmap ----
    fig1, ax = plt.subplots(figsize=(7.2, 5.4), layout="constrained")
    grid_val = np.array([[reg[m], slp[m]] for m in rows])          # displayed metric
    grid_delta = np.array([[reg[m] - reg_floor, slp[m] - slp_floor] for m in rows])
    norm = TwoSlopeNorm(vmin=-VLIM, vcenter=0.0, vmax=VLIM)
    disp = np.ma.masked_invalid(grid_delta)
    ax.imshow(disp, cmap=CMAP, norm=norm, aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["node regression\n(10-shot ρ)", "static link pred.\n(0-shot AUC)"], fontsize=11)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{LABELS[m][0]}" for m in rows], fontsize=11)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # separator above the floor rows
    ax.axhline(len(STRATS) - 0.5, color="#444441", lw=1.0)

    for i, m in enumerate(rows):
        for j, (val, dv) in enumerate(zip([reg[m], slp[m]], [grid_delta[i, 0], grid_delta[i, 1]])):
            if not np.isfinite(val):
                txt, sub, col = ("pending", "", "#888780") if (m == "features_only" and j == 0) else ("—", "", "#B4B2A9")
                ax.text(j, i, txt, ha="center", va="center", fontsize=11, color=col)
                continue
            dark = abs(dv) > 0.09
            col = "white" if dark else "#2C2C2A"
            ax.text(j, i - 0.13, f"{val:.2f}", ha="center", va="center", fontsize=15, color=col, fontweight="bold")
            tag = "floor" if m == FLOOR else f"{dv:+.2f}"
            ax.text(j, i + 0.22, tag, ha="center", va="center", fontsize=9.5,
                    color=col if m != FLOOR else "#5F5E5A")

    ax.set_title("Frozen-probe cross-task matrix — margin over random-encoder floor",
                 fontsize=12.5, pad=10)
    fig1.savefig(OUT / "probe_matrix_heatmap.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig1.savefig(OUT / "probe_matrix_heatmap.pdf", bbox_inches="tight", facecolor="white")

    # ---- fig 2: diverging Δ-over-floor bars ----
    fig2, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), layout="constrained")
    tasks = [("node regression", reg, reg_floor, "10-shot Spearman ρ"),
             ("static link prediction", slp, slp_floor, "0-shot ROC-AUC")]
    for ax, (title, data, floor, unit) in zip(axes, tasks):
        deltas = [data[m] - floor for m in STRATS]
        labs = [LABELS[m][0] for m in STRATS]
        y = np.arange(len(STRATS))[::-1]
        colors = ["#378ADD" if d > 0 else "#E24B4A" for d in deltas]
        ax.barh(y, deltas, color=colors, height=0.62, zorder=3)
        ax.axvline(0, color="#444441", lw=1.1, zorder=4)
        for yi, d in zip(y, deltas):
            ax.text(d + (0.006 if d >= 0 else -0.006), yi, f"{d:+.2f}",
                    va="center", ha="left" if d >= 0 else "right", fontsize=10.5,
                    color="#0C447C" if d > 0 else "#A32D2D")
        ax.set_yticks(y)
        ax.set_yticklabels(labs, fontsize=11)
        ax.set_xlim(-0.16, 0.27)
        ax.set_title(f"{title}\n(Δ vs random encoder, {unit})", fontsize=11.5)
        ax.tick_params(length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_xlabel("0 = random-encoder floor", fontsize=9.5, color="#5F5E5A")
    fig2.suptitle("Only single-source NM beats an untrained encoder — and only on static LP",
                  fontsize=12.5)
    fig2.savefig(OUT / "probe_matrix_delta_bars.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig2.savefig(OUT / "probe_matrix_delta_bars.pdf", bbox_inches="tight", facecolor="white")

    print(f"\nwrote 4 files to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
