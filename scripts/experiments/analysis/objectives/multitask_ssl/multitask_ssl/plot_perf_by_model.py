#!/usr/bin/env python3
"""Performance-by-model scatter for the mixed-objective arms.

x = model (arm)     — NM / CL / FP / MIX
y = performance     — the task's headline metric (Spearman ρ / ROC-AUC)
shape = test graph  — one marker per dataset (fixed across tasks)
color = target      — for regression (followers/statuses/account-age);
                      classification & static-LP have no sub-target, so they use a
                      single task color and dodge by test graph.

One point per test-split eval instance (regression: model×dataset×target;
cls/slp: model×dataset). Encoding maps are module-level and fixed so a dataset keeps
its shape and a task keeps its color across every figure.

    python .../plot_perf_by_model.py --task regression
    python .../plot_perf_by_model.py --task classification
    python .../plot_perf_by_model.py --task static_link_prediction
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())
PLOT = REPO / "scripts/experiments/analysis/objectives/multitask_ssl/multitask_ssl"

MODELS = ["NM", "CL", "FP", "MIX"]
MODEL_SUB = {"NM": "neighbor\nmatching", "CL": "contrastive", "FP": "feature\nrecon", "MIX": "rotation"}

TASK = {
    "regression": dict(csv="node_regression/data/node_regression.csv", metric="spearman",
                       color="#F28E2B", ref=0.0, ref_txt="no signal (ρ=0)",
                       ylab="node-regression Spearman ρ", color_by="target"),
    "classification": dict(csv="node_classification/data/node_classification.csv", metric="roc_auc",
                           color="#4E79A7", ref=0.5, ref_txt="chance (AUC=0.5)",
                           ylab="node-classification ROC-AUC", color_by=None),
    "static_link_prediction": dict(csv="static_link_prediction/data/static_link_prediction.csv", metric="roc_auc",
                                    color="#59A14F", ref=0.5, ref_txt="chance (AUC=0.5)",
                                    ylab="static-link-prediction ROC-AUC", color_by=None),
}

# color = regression target
TARGET_COLOR = {"followers_count": "#4E79A7", "statuses_count": "#F28E2B", "account_age_days": "#59A14F"}
TARGET_DISP = {"followers_count": "followers", "statuses_count": "statuses", "account_age_days": "account age"}
TARGET_ORDER = ["followers_count", "statuses_count", "account_age_days"]

# shape = test graph (fixed globally)
DATASET_SHAPE = {"midterm": "o", "ukr_rus_twitter": "s", "covid19_twitter": "^",
                 "twibot20": "D", "election2020": "P"}
DATASET_DISP = {"midterm": "midterm", "ukr_rus_twitter": "ukr_rus", "covid19_twitter": "covid19",
                "twibot20": "twibot20", "election2020": "election2020"}
DATASET_ORDER = ["midterm", "ukr_rus_twitter", "covid19_twitter", "twibot20", "election2020"]


def load(task):
    cfg = TASK[task]
    rows = []
    with (PLOT / cfg["csv"]).open() as fh:
        for r in csv.DictReader(fh):
            if r.get("split") != "test" or r.get(cfg["metric"]) in ("", None):
                continue
            rows.append((r["model"], r["dataset"], r.get("target", ""), float(r[cfg["metric"]])))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="regression", choices=list(TASK))
    ap.add_argument("--dodge", type=float, default=0.14)
    ap.add_argument("--size", type=float, default=70.0)
    ap.add_argument("--outdir", default=str(HERE / "figures"))
    args = ap.parse_args()
    cfg = TASK[args.task]
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rows = load(args.task)

    datasets = [ds for ds in DATASET_ORDER if any(r[1] == ds for r in rows)]
    color_by = cfg["color_by"]
    # sub-columns to dodge by: targets (regression) or datasets (cls/slp)
    if color_by == "target":
        groups = [t for t in TARGET_ORDER if any(r[2] == t for r in rows)]
    else:
        groups = datasets
    offs = dict(zip(groups, np.linspace(-args.dodge, args.dodge, len(groups)) if len(groups) > 1 else [0.0]))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.edgecolor": "#9aa0a6", "axes.linewidth": 1.0, "figure.dpi": 120})
    fig, ax = plt.subplots(figsize=(8.8, 6.2))

    ax.axhline(cfg["ref"], color="#b9bec4", lw=1.2, ls=(0, (5, 4)), zorder=1)
    ax.text(len(MODELS) - 0.53, cfg["ref"] + 0.006, cfg["ref_txt"], color="#9aa0a6",
            fontsize=9, ha="right", va="bottom")
    for xi in range(len(MODELS)):
        ax.axvspan(xi - 0.5, xi + 0.5, color="#f6f7f9" if xi % 2 else "#ffffff", zorder=0)

    def xy(model, ds, t):
        xi = MODELS.index(model)
        g = t if color_by == "target" else ds
        return xi + offs.get(g, 0.0)

    for t in (groups if color_by == "target" else [None]):
        for ds in datasets:
            xs, ys = [], []
            for (m, d2, t2, v) in rows:
                if d2 != ds:
                    continue
                if color_by == "target" and t2 != t:
                    continue
                xs.append(xy(m, ds, t2)); ys.append(v)
            if not xs:
                continue
            c = TARGET_COLOR[t] if color_by == "target" else cfg["color"]
            ax.scatter(xs, ys, marker=DATASET_SHAPE[ds], s=args.size, c=c,
                       edgecolors="white", linewidths=0.9, zorder=5, alpha=0.95)

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([f"{m}\n{MODEL_SUB[m]}" for m in MODELS], fontsize=11)
    ax.set_xlim(-0.5, len(MODELS) - 0.5)
    ax.set_xlabel("model (SSL objective)", fontsize=12, labelpad=8)
    ax.set_ylabel(f"performance  —  {cfg['ylab']}", fontsize=12)
    ax.set_title(f"{args.task.replace('_', ' ').title()} transfer by model", fontsize=15, fontweight="bold", pad=26)
    color_txt = "color = target" if color_by == "target" else "color = task"
    ax.text(0.5, 1.028, f"frozen-encoder transfer · shape = test graph · {color_txt} · 1 seed",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="#5f6368")

    ax.grid(True, axis="y", color="#eceef1", lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    shape_handles = [Line2D([0], [0], marker=DATASET_SHAPE[ds], ls="", markerfacecolor="#59607a",
                            markeredgecolor="white", markeredgewidth=0.9, markersize=9,
                            label=DATASET_DISP[ds]) for ds in datasets]
    leg1 = ax.legend(handles=shape_handles, title="test graph (shape)", loc="best",
                     fontsize=9.5, frameon=True, framealpha=0.95, edgecolor="#e0e2e5",
                     borderpad=0.6, handletextpad=0.4)
    leg1.get_title().set_fontsize(9); leg1.get_title().set_color("#5f6368")
    ax.add_artist(leg1)

    if color_by == "target":
        color_handles = [Line2D([0], [0], marker="s", ls="", markerfacecolor=TARGET_COLOR[t],
                                markeredgecolor="white", markeredgewidth=0.9, markersize=11,
                                label=TARGET_DISP[t]) for t in groups]
        leg2 = ax.legend(handles=color_handles, title="target (color)", loc="upper right",
                         fontsize=9.5, frameon=True, framealpha=0.95, edgecolor="#e0e2e5",
                         borderpad=0.6, handletextpad=0.4)
        leg2.get_title().set_fontsize(9); leg2.get_title().set_color("#5f6368")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = outdir / f"perf_by_model_{args.task}.{ext}"
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
