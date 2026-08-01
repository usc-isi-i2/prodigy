#!/usr/bin/env python3
"""Build the saturation long table and the curve figure from the shared per-task CSVs.

Reads the append-only CSVs written by
``scripts/harness/benchmark_tasks/parse_benchmark_eval_logs.py``, keeps only the rows
this experiment produced (model keys ``sat_<arm>_s<step>``), and emits:

    data/pretrain_saturation_long.csv    one row per (arm, step, task, dataset, target)
    data/pretrain_saturation_wide.csv    arm x step, mean of the primary metric per task
    figures/pretrain_saturation.png      the curve

Run with the Homebrew Python 3.11 (has pandas/matplotlib); the repo's conda env is for
training, not plotting.

    /opt/homebrew/bin/python3.11 build_tables_and_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent

# Arm order is the experiment's story order: broadest corpus first, then the two
# single-source rungs. Colour follows the ARM, never its rank on the plot, so a filtered
# or reordered figure never repaints a series.
ARM_ORDER = ["all8", "ukr", "covid"]
ARM_LABEL = {
    "all8": "all8 (8-source merge)",
    "ukr":  "ukr (single source)",
    "covid": "covid (single source)",
}
# Validated categorical slots 1-3 (light surface): blue / orange / aqua.
# scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light --pairs all -> all PASS,
# with a contrast WARN on the aqua that direct labels relieve.
ARM_COLOR = {"all8": "#2a78d6", "ukr": "#eb6834", "covid": "#1baf7a"}

INK, INK_MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dd"

TASKS = [
    # (task key, source csv, metric column, axis label, panel title)
    ("classification", "node_classification/data/node_classification.csv", "roc_auc",
     "ROC-AUC", "Node classification  (4 graphs, 10-shot)"),
    ("regression", "node_regression/data/node_regression.csv", "spearman",
     "Spearman", "Node regression — VOID, superseded (see probe_regression_curves.png)"),
]


def load(csv_rel: str, metric: str, task: str) -> pd.DataFrame:
    df = pd.read_csv(ANALYSIS / csv_rel)
    df = df[df["model"].astype(str).str.startswith("sat_")].copy()
    keys = df["model"].str.extract(r"^sat_(?P<arm>[a-z0-9]+)_s(?P<step>\d+)$")
    df["arm"] = keys["arm"]
    # int, not the zero-padded string: the padding exists so a lexical sort is a numeric
    # sort, but plotting on a log axis needs the number.
    df["step"] = keys["step"].astype(int)
    df["task"] = task
    df["metric"] = metric
    df["value"] = df[metric]
    cols = ["arm", "step", "task", "dataset", "target", "metric", "value", "shots", "model"]
    return df[[c for c in cols if c in df.columns]]


def classification_sigma() -> float | None:
    """Single-run sigma of ROC-AUC, from the paired replicate runs. None if absent."""
    rep_path = HERE / "data" / "classification_replicates.csv"
    if not rep_path.is_file():
        return None
    rep = pd.read_csv(rep_path)
    orig = pd.read_csv(ANALYSIS / TASKS[0][1])
    key = ["dataset", "shots"]
    rep = rep.assign(k=rep.model.str.replace("^rep_", "", regex=True))
    orig = orig[orig.model.astype(str).str.startswith("sat_")].assign(
        k=lambda d: d.model.str.replace("^sat_", "", regex=True))
    m = rep.merge(orig, on=["k"] + key, suffixes=("_rep", "_orig"))
    if m.empty:
        return None
    diffs = (m.roc_auc_rep - m.roc_auc_orig).to_numpy()
    return float(diffs.std(ddof=1) / (2 ** 0.5))


def step0_anchor() -> pd.DataFrame | None:
    """Step-0 (untrained) values per cell. All three arms share ONE t=0 encoder --
    verified byte-identical across arms -- so this is a single reference level per cell,
    not a per-arm curve point. It also cannot be drawn ON a log x-axis (log(0)), which is
    the other reason it is a horizontal line rather than a leftmost marker."""
    f = HERE / "data" / "step0_anchor.csv"
    return pd.read_csv(f) if f.is_file() else None


def main() -> int:
    frames = [load(csv, metric, task) for task, csv, metric, _, _ in TASKS]
    long = pd.concat(frames, ignore_index=True).sort_values(
        ["task", "arm", "step", "dataset", "target"])

    expected = 3 * 6 * (4 + 4 * 2)   # arms x steps x (classification + regression cells)
    if len(long) != expected:
        print(f"[warn] {len(long)} rows, expected {expected} -- an eval cell is missing",
              file=sys.stderr)

    # Run-to-run uncertainty, measured: two independent runs of the identical config
    # scored on the same cells (data/classification_replicates.csv). sigma for ONE run is
    # the paired-difference sigma over sqrt(2); a band drawn at +/-sigma is therefore what
    # a single curve is worth, not the gap between two of them.
    band = classification_sigma()
    step0 = step0_anchor()

    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    long.to_csv(HERE / "data" / "pretrain_saturation_long.csv", index=False)

    wide = (long.pivot_table(index=["task", "step"], columns="arm", values="value",
                             aggfunc="mean")
                .reindex(columns=ARM_ORDER).round(4))
    wide.to_csv(HERE / "data" / "pretrain_saturation_wide.csv")
    print(wide.to_string())

    # --- figure: two panels, NOT two y-axes. The metrics have different scales and
    # meanings, so they get separate panels sharing only the x axis.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, (task, _, _, ylab, title) in zip(axes, TASKS):
        sub = long[long["task"] == task]
        ends = {}
        for arm in ARM_ORDER:
            s = (sub[sub["arm"] == arm].groupby("step")["value"].mean().sort_index())
            if task == "classification" and band:
                ax.fill_between(s.index, s.values - band, s.values + band,
                                color=ARM_COLOR[arm], alpha=0.16, linewidth=0, zorder=2)
            ax.plot(s.index, s.values, color=ARM_COLOR[arm], linewidth=2,
                    marker="o", markersize=5.5, markeredgecolor="white",
                    markeredgewidth=1.2, label=ARM_LABEL[arm], zorder=3,
                    solid_capstyle="round")
            ends[arm] = (s.index[-1], s.values[-1])

        ax.set_xscale("log")
        ax.set_xlabel("pretraining steps (log)", color=INK_MUTED, fontsize=9)
        ax.set_ylabel(ylab, color=INK_MUTED, fontsize=9)
        ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=10)
        ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=8.5)
        # CLASSIFICATION ONLY. The step-0 regression anchor is a PROBE Spearman; the
        # right panel plots the void episodic Spearman. Drawing one on the other would
        # put two different measurements on one axis. The probe's own step-0 line lives
        # in probe_regression_curves.png, where it belongs.
        if step0 is not None and task == "classification":
            lvl = step0[step0.task == task]["value"].mean()
            ax.axhline(lvl, color=INK_MUTED, linewidth=1.3, linestyle=(0, (5, 3)), zorder=1)
            ax.annotate(f"untrained encoder (step 0) = {lvl:.3f}", (0.985, lvl),
                        xycoords=("axes fraction", "data"), xytext=(0, 5),
                        textcoords="offset points", fontsize=7.5, color=INK_MUTED,
                        va="bottom", ha="right")
        ax.set_xlim(80, 90000)   # headroom for the direct labels
        if task == "regression":
            ax.axhline(0, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)

        # Direct labels at each line's last point -- also the relief the palette validator
        # requires for the low-contrast aqua series. Minimum separation is keyed to the
        # RENDERED axis range, not to the spread of the endpoints: label collision is a
        # function of text height on the page. On the classification panel the endpoints
        # sit 0.0035 apart while the labels need ~0.010, so an endpoint-relative
        # threshold silently declines to nudge exactly where nudging is needed.
        y0, y1 = ax.get_ylim()
        min_gap = 0.052 * (y1 - y0)
        placed: list[float] = []
        for arm in sorted(ARM_ORDER, key=lambda a: ends[a][1], reverse=True):
            x, y_label = ends[arm]
            while any(abs(y_label - p) < min_gap for p in placed):
                y_label -= min_gap
            placed.append(y_label)
            ax.annotate(arm, xy=(x, y_label), xytext=(12, 0),
                        textcoords="offset points", color=ARM_COLOR[arm],
                        fontsize=9, va="center", ha="left", zorder=4,
                        annotation_clip=False)

    # lower LEFT: the step-0 reference label now occupies the lower right.
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK)
    # Title states only what both panels support. Classification saturates by ~500 steps;
    # regression does not saturate at anything, it never leaves the noise around zero, so
    # a single "saturation is early" headline would misdescribe the right-hand panel.
    fig.suptitle("Downstream transfer vs pretraining budget",
                 color=INK, fontsize=12.5, x=0.008, ha="left", y=1.0)
    fig.text(0.008, 0.935,
             "Classification reaches its plateau by ~500 steps: the rise is 16x the run-to-run "
             "noise, the plateau 1.1x it — i.e. flat to within measurement error.",
             color=INK_MUTED, fontsize=9, ha="left")
    fig.text(0.008, -0.10,
             "Steps 100/500 from dense retrains; 1000+ spliced from the original runs.\n"
             "Shaded band on the left = measured single-run sigma (0.012) from two independent "
             "runs of the identical config.\n"
             "RIGHT PANEL IS VOID — it plots the episodic regression eval, whose regression_head "
             "is random and never fitted; valid regression is in probe_regression_curves.png.",
             color=INK_MUTED, fontsize=8, ha="left", linespacing=1.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = HERE / "figures" / "pretrain_saturation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {out}")
    print(f"wrote {HERE / 'data' / 'pretrain_saturation_long.csv'} ({len(long)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
