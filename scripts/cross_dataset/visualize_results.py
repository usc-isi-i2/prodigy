"""
Cross-dataset experiment visualizations.
Usage: python visualize_results.py [--csv results.csv] [--out plots/]

All metrics are ROC-AUC (test_roc_auc) for consistency across NM, LP, and PL tasks.
Rows with both test_acc and test_roc_auc missing are skipped (pending eval jobs).

Produces 9 figures:
  1. main_results.png          — ROC-AUC vs accuracy at 10-shot, grouped bars per regime
  2. shot_curves.png           — ROC-AUC vs shots per (eval_dataset, eval_task), lines per regime
  3. heatmap.png               — regime x eval-task grid at 10-shot (ROC-AUC)
  4. heatmap_acc.png           — same, using accuracy
  5. heatmap_flat.png          — single flat heatmap, all shots (ROC-AUC)
  6. heatmap_flat_acc.png      — same, using accuracy
  7. cross_task_delta_nm.png   — cross-task delta vs NM->NM specialist on NM eval, by dataset
  8. cross_task_delta_lp.png   — cross-task delta vs LP->LP specialist on LP eval, by dataset
  9. cross_task_delta_pl.png   — cross-task delta vs avg baselines on PL eval (no PL specialist), by dataset

Datasets: covid (NM + LP + PL), midterm (NM + LP + PL), ukr_rus (NM + LP + PL)
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# ── constants ──────────────────────────────────────────────────────────────────
REGIME_ORDER  = ["NM->NM", "LP->LP", "NM->LP", "LP->NM"]
REGIME_COLORS = {"NM->NM": "#4C72B0", "LP->LP": "#DD8452", "NM->LP": "#55A868", "LP->NM": "#C44E52"}
SHOT_ORDER    = [1, 5, 10]
TASK_ORDER    = ["NM", "LP", "PL"]
DS_ORDER      = ["covid", "midterm", "ukr_rus"]

TASK_METRIC   = {"NM": "test_roc_auc", "LP": "test_roc_auc", "PL": "test_roc_auc"}

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams["figure.dpi"] = 130


# ── helpers ────────────────────────────────────────────────────────────────────
def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["shots"] = df["shots"].astype(int)
    return df


def exp_label(row) -> str:
    return f"Exp {int(row['experiment'])}\n{row['training_regime']}"


def best_metric(task: str, row) -> float:
    """Return the most meaningful metric for each task."""
    col = TASK_METRIC[task]
    return row[col]


# ── Figure 1: Main results — ROC-AUC vs Accuracy at 10-shot ───────────────────
# Layout: rows = eval_task, cols = eval_dataset
# Within each cell: x = regime, 2 bars per regime (ROC-AUC solid, accuracy hatched)
def plot_main_results(df: pd.DataFrame, out_dir: str):
    sub      = df[df["shots"] == 10].copy()
    tasks    = [t for t in TASK_ORDER   if t in sub["eval_task"].unique()]
    datasets = [d for d in DS_ORDER     if d in sub["eval_dataset"].unique()]
    regimes  = [r for r in REGIME_ORDER if r in sub["training_regime"].unique()]

    bar_w   = 0.32
    offsets = [-bar_w / 2, bar_w / 2]   # ROC-AUC left, accuracy right
    x       = np.arange(len(regimes))

    fig, axes = plt.subplots(
        len(tasks), len(datasets),
        figsize=(4.5 * len(datasets), 3.5 * len(tasks)),
        sharey="row", sharex=True,
    )
    if len(tasks) == 1:
        axes = [axes]
    if len(datasets) == 1:
        axes = [[ax] for ax in axes]

    for row_i, task in enumerate(tasks):
        for col_i, ds in enumerate(datasets):
            ax = axes[row_i][col_i]
            cell = sub[(sub["eval_task"] == task) & (sub["eval_dataset"] == ds)]

            if cell.empty:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="grey", fontsize=13)
                _style_ax(ax, row_i, col_i, task, ds)
                continue

            for xi, regime in enumerate(regimes):
                r = cell[cell["training_regime"] == regime]
                if r.empty:
                    continue
                rep = r.iloc[0]
                regime_label = (
                    f"{rep['pretrain_dataset']} ({rep['pretrain_task']}) -> "
                    f"{rep['finetune_dataset']} ({rep['finetune_task']})"
                )
                roc = r["test_roc_auc"].mean()
                acc = r["test_acc"].mean()

                for val, offset, hatch in [
                    (roc, offsets[0], None),
                    (acc, offsets[1], "///"),
                ]:
                    ax.bar(xi + offset, val, width=bar_w,
                           color=REGIME_COLORS[regime], hatch=hatch,
                           edgecolor="white", linewidth=0.6,
                           label=regime_label)
                    if not np.isnan(val):
                        ax.text(xi + offset, val + 0.005, f"{val:.2f}",
                                ha="center", va="bottom", fontsize=6.5)

            ax.set_xticks(x)
            ax.set_xticklabels(regimes, fontsize=8, rotation=15, ha="right")
            _style_ax(ax, row_i, col_i, task, ds)

    seen, regime_handles, regime_labels = set(), [], []
    for row in axes:
        for ax in row:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen and not l.startswith("_"):
                    seen.add(l)
                    regime_handles.append(h)
                    regime_labels.append(l)

    metric_patches = [
        mpatches.Patch(facecolor="grey", hatch=None,  label="ROC-AUC"),
        mpatches.Patch(facecolor="grey", hatch="///", label="Accuracy"),
    ]

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.legend(regime_handles, regime_labels, title="Regime",
               loc="lower left", bbox_to_anchor=(0.01, 0.01),
               frameon=True, fontsize=9, title_fontsize=10, ncol=3)
    fig.legend(handles=metric_patches, title="Metric",
               loc="lower right", bbox_to_anchor=(0.99, 0.01),
               frameon=True, fontsize=10, title_fontsize=10, ncol=2)
    fig.suptitle("ROC-AUC vs Accuracy at 10-shot by training regime", fontsize=14, y=1.01)
    _save(fig, out_dir, "main_results.png")


def _style_ax(ax, row_i, col_i, task_label, ds_label):
    if row_i == 0:
        ax.set_title(f"Eval: {ds_label}", fontsize=11, fontweight="bold")
    if col_i == 0:
        ax.set_ylabel(f"{task_label}\nROC-AUC", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))


# ── Figure 2: Shot learning curves ────────────────────────────────────────────
# Rows = eval_task, cols = eval_dataset
# Lines = one per training_regime (averaged across experiments in that regime)
def plot_shot_curves(df: pd.DataFrame, out_dir: str):
    tasks    = [t for t in TASK_ORDER if t in df["eval_task"].unique()]
    datasets = [d for d in DS_ORDER if d in df["eval_dataset"].unique()]

    fig, axes = plt.subplots(
        len(tasks), len(datasets),
        figsize=(4 * len(datasets), 3.2 * len(tasks) + 2.5),
        sharey="row", sharex=True,   # same y scale within each row, auto-ranged per row
    )
    if len(tasks) == 1:
        axes = [axes]
    if len(datasets) == 1:
        axes = [[ax] for ax in axes]

    for row_i, task in enumerate(tasks):
        metric = TASK_METRIC[task]
        for col_i, ds in enumerate(datasets):
            ax = axes[row_i][col_i]
            cell = df[(df["eval_task"] == task) & (df["eval_dataset"] == ds)]

            if cell.empty:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
                _style_curve_ax(ax, task, row_i, col_i, ds, len(tasks))
                continue

            # Build chain label once per regime from the first matching row
            for regime in REGIME_ORDER:
                r = cell[cell["training_regime"] == regime]
                if r.empty:
                    continue
                rep = r.iloc[0]
                line_label = (
                    f"{rep['pretrain_dataset']} ({rep['pretrain_task']}) -> "
                    f"{rep['finetune_dataset']} ({rep['finetune_task']})"
                )

                agg = r.groupby("shots")[metric].mean().reindex(SHOT_ORDER)
                # Individual experiment lines (faint)
                for _, eg in r.groupby("experiment"):
                    vals = eg.set_index("shots")[metric].reindex(SHOT_ORDER)
                    ax.plot(SHOT_ORDER, vals.values,
                            color=REGIME_COLORS[regime], alpha=0.25,
                            linewidth=1, linestyle="--")
                # Regime mean (bold)
                ax.plot(SHOT_ORDER, agg.values,
                        color=REGIME_COLORS[regime], linewidth=2.5,
                        marker="o", markersize=6, label=line_label)

            _style_curve_ax(ax, task, row_i, col_i, ds, len(tasks))

    # Collect unique labels from all axes for a shared legend
    seen, handles, labels = set(), [], []
    for row in axes:
        for ax in row:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    seen.add(l)
                    handles.append(h)
                    labels.append(l)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    fig.legend(handles, labels, title="Training regime",
               ncol=3, fontsize=11, title_fontsize=12,
               loc="lower center",
               bbox_to_anchor=(0.0, 0.0, 1.0, 0.20),
               bbox_transform=fig.transFigure,
               mode="expand", frameon=True)
    fig.suptitle("Few-shot learning curves by training regime", fontsize=14, y=1.01)
    _save(fig, out_dir, "shot_curves.png")


def _style_curve_ax(ax, task, row_i, col_i, ds_label, n_tasks):
    if row_i == 0:
        ax.set_title(f"Eval: {ds_label}", fontsize=11, fontweight="bold")
    if col_i == 0:
        ax.set_ylabel(f"{task}\nROC-AUC", fontsize=10)
    if row_i == n_tasks - 1:
        ax.set_xlabel("shots")
    ax.set_xticks(SHOT_ORDER)


# ── Figure 3a: Heatmap (10-shot, 3 panels) ────────────────────────────────────
# Three dense panels side by side — one per eval_dataset.
# Rows = training regime, cols = eval tasks. No empty cells.
def plot_heatmap(df: pd.DataFrame, out_dir: str, metric: str = "test_roc_auc"):
    suffix  = "" if metric == "test_roc_auc" else "_acc"
    mlabel  = "ROC-AUC" if metric == "test_roc_auc" else "Accuracy"

    sub = df[df["shots"] == 10].copy()
    sub["metric_val"] = sub[metric]

    datasets = [d for d in DS_ORDER if d in sub["eval_dataset"].unique()]
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(3.5 * len(datasets) + 1, 3.8),
                             gridspec_kw={"wspace": 0.4})
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        panel = sub[sub["eval_dataset"] == ds]
        tasks_here = [t for t in TASK_ORDER if t in panel["eval_task"].unique()]

        rep = panel.iloc[0]
        chain = f"{rep['pretrain_dataset']} → {rep['finetune_dataset']}"

        pivot = (panel.groupby(["training_regime", "eval_task"])["metric_val"]
                      .mean()
                      .unstack("eval_task")
                      .reindex(index=REGIME_ORDER, columns=tasks_here))

        sns.heatmap(
            pivot.astype(float), ax=ax,
            annot=True, fmt=".2f", annot_kws={"size": 11, "weight": "bold"},
            cmap="RdYlGn", vmin=0.4, vmax=1.0,
            linewidths=1.5, linecolor="white",
            cbar=False,
        )

        for tick_label in ax.get_yticklabels():
            regime = tick_label.get_text()
            tick_label.set_color(REGIME_COLORS.get(regime, "black"))
            tick_label.set_fontweight("bold")

        ax.set_title(f"Eval: {ds}\n{chain}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("eval task", fontsize=9)
        ax.set_ylabel("training regime" if ax == axes[0] else "", fontsize=9)
        ax.tick_params(axis="x", rotation=0, labelsize=10)
        ax.tick_params(axis="y", rotation=0, labelsize=10)

    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin=0.4, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.02)
    cbar.set_label(mlabel, fontsize=9)
    fig.suptitle(f"10-shot performance by training regime and eval target ({mlabel})",
                 fontsize=13, y=1.04)
    _save(fig, out_dir, f"heatmap{suffix}.png")


# ── Figure 3b: Heatmap flat (all shots) ───────────────────────────────────────
# Single heatmap: rows = regime × shots, cols = eval_dataset × eval_task.
def plot_heatmap_flat(df: pd.DataFrame, out_dir: str, metric: str = "test_roc_auc"):
    suffix = "" if metric == "test_roc_auc" else "_acc"
    df = df.copy()

    row_keys = [(regime, shots) for regime in REGIME_ORDER for shots in SHOT_ORDER
                if regime in df["training_regime"].values]
    col_keys = [(ds, task) for ds in DS_ORDER for task in TASK_ORDER
                if ((df["eval_dataset"] == ds) & (df["eval_task"] == task)).any()]

    # Build row labels: include pretrain->finetune datasets from the first matching row
    row_labels = []
    for regime, shots in row_keys:
        r = df[df["training_regime"] == regime]
        if not r.empty:
            rep = r.iloc[0]
            row_labels.append(
                f"{rep['pretrain_dataset']} ({rep['pretrain_task']}) -> "
                f"{rep['finetune_dataset']} ({rep['finetune_task']})  {shots}shot"
            )
        else:
            row_labels.append(f"{regime}  {shots}shot")

    col_labels = [f"{d}\n{t}" for d, t in col_keys]

    data = np.full((len(row_keys), len(col_keys)), np.nan)
    for ri, (regime, shots) in enumerate(row_keys):
        for ci, (ds, task) in enumerate(col_keys):
            val = df[(df["training_regime"] == regime) &
                     (df["shots"] == shots) &
                     (df["eval_dataset"] == ds) &
                     (df["eval_task"] == task)][metric]
            if not val.empty:
                data[ri, ci] = val.mean()

    pivot = pd.DataFrame(data, index=row_labels, columns=col_labels)

    fig, ax = plt.subplots(figsize=(len(col_keys) * 1.1 + 1, len(row_keys) * 0.85 + 0.5))
    sns.heatmap(
        pivot, ax=ax,
        annot=True, fmt=".2f", annot_kws={"size": 9, "weight": "bold"},
        cmap="RdYlGn", vmin=0.4, vmax=1.0,
        linewidths=0.8, linecolor="white",
        cbar_kws={"shrink": 0.6},
    )

    # Colour by regime using row_keys (same order as pivot rows)
    for tick_label, (regime, _) in zip(ax.get_yticklabels(), row_keys):
        tick_label.set_color(REGIME_COLORS.get(regime, "black"))
        tick_label.set_fontweight("bold")

    for i in range(1, len(REGIME_ORDER)):
        ax.axhline(i * len(SHOT_ORDER), color="black", linewidth=1.5)

    ax.tick_params(axis="x", rotation=0, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.tight_layout()
    _save(fig, out_dir, f"heatmap_flat{suffix}.png")


# ── Figure 4: Cross-task delta ────────────────────────────────────────────────
# Do NM->LP and LP->NM beat the same-task baselines (NM->NM and LP->LP)?
# One figure per eval task; bars grouped by dataset × regime, coloured by shots.
def plot_cross_task_delta(df: pd.DataFrame, out_dir: str):
    # Baseline: use the specialist whose training task matches the eval task.
    # PL has no specialist → fall back to avg(NM->NM, LP->LP).
    TASK_SPECIALIST = {"NM": "NM->NM", "LP": "LP->LP"}
    CROSS_REGIMES   = ["NM->LP", "LP->NM"]

    rows = []
    for (ds, task, shots), g in df.groupby(["eval_dataset", "eval_task", "shots"]):
        metric = TASK_METRIC[task]

        if task in TASK_SPECIALIST:
            baseline_val = g[g["training_regime"] == TASK_SPECIALIST[task]][metric].mean()
            baseline_lbl = TASK_SPECIALIST[task]
        else:
            baseline_val = g[g["training_regime"].isin(["NM->NM", "LP->LP"])][metric].mean()
            baseline_lbl = "avg(NM->NM, LP->LP)"

        for regime in CROSS_REGIMES:
            cross_val = g[g["training_regime"] == regime][metric].mean()
            if pd.notna(baseline_val) and pd.notna(cross_val):
                rows.append({"eval_dataset": ds, "eval_task": task, "shots": shots,
                             "regime": regime, "delta": cross_val - baseline_val,
                             "baseline": baseline_lbl, "label": ds})

    if not rows:
        print("Not enough data for cross-task delta plot — skipping.")
        return

    delta_df = pd.DataFrame(rows)
    tasks_present = [t for t in TASK_ORDER if t in delta_df["eval_task"].unique()]

    task_titles = {"NM": "Neighbour Matching (NM)", "LP": "Link Prediction (LP)", "PL": "Political Leaning (PL)"}
    shot_alphas = {1: 0.50, 5: 0.70, 10: 0.90}

    for task in tasks_present:
        sub      = delta_df[delta_df["eval_task"] == task]
        datasets = [d for d in DS_ORDER if d in sub["eval_dataset"].unique()]
        regimes  = [r for r in CROSS_REGIMES if r in sub["regime"].unique()]
        n_ds     = len(datasets)
        n_reg    = len(regimes)

        # Each dataset gets a group; within the group, bars are ordered regime × shot.
        n_bars   = n_reg * len(SHOT_ORDER)
        bar_h    = 0.18
        group_h  = n_bars * bar_h + 0.1   # height per dataset group
        y_centers = np.arange(n_ds) * group_h

        fig, ax = plt.subplots(figsize=(9, max(3.0, n_ds * group_h + 1.0)))

        for ds_i, ds in enumerate(datasets):
            ds_sub = sub[sub["eval_dataset"] == ds]
            bar_idx = 0
            for regime in regimes:
                reg_sub = ds_sub[ds_sub["regime"] == regime]
                for shots in SHOT_ORDER:
                    row = reg_sub[reg_sub["shots"] == shots]
                    if row.empty:
                        bar_idx += 1
                        continue
                    v      = row["delta"].iloc[0]
                    y      = y_centers[ds_i] + (bar_idx - n_bars / 2 + 0.5) * bar_h
                    color  = REGIME_COLORS[regime]
                    alpha  = shot_alphas[shots]
                    ax.barh(y, v, height=bar_h * 0.9, color=color, alpha=alpha)
                    xpos = v + 0.001 if v >= 0 else v - 0.001
                    ha   = "left" if v >= 0 else "right"
                    ax.text(xpos, y, f"{v:+.3f}", va="center", ha=ha, fontsize=7.5)
                    bar_idx += 1

        baseline_lbl = sub["baseline"].iloc[0]
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
        ax.set_yticks(y_centers)
        ax.set_yticklabels(datasets, fontsize=11)
        ax.set_xlabel(f"cross-task regime  minus  {baseline_lbl}", fontsize=10)
        ax.set_title(f"{task_titles[task]}\nPositive = cross-task beats {baseline_lbl}",
                     fontsize=12)
        ax.invert_yaxis()

        # Legend: regime colour + shot alpha
        legend_handles = []
        for regime in regimes:
            legend_handles.append(mpatches.Patch(color=REGIME_COLORS[regime], label=regime))
        for shots in SHOT_ORDER:
            legend_handles.append(
                mpatches.Patch(facecolor="grey", alpha=shot_alphas[shots], label=f"{shots}-shot"))
        ax.legend(handles=legend_handles, fontsize=9, loc="best",
                  title="regime / shots", title_fontsize=9)

        fig.tight_layout()
        _save(fig, out_dir, f"cross_task_delta_{task.lower()}.png")


# ── utility ────────────────────────────────────────────────────────────────────
def _save(fig, out_dir: str, name: str):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results.csv")
    parser.add_argument("--out", default="plots")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load(args.csv)
    df = df.dropna(subset=["test_acc", "test_roc_auc"], how="all")

    if df.empty:
        print("No data found — fill in test_acc / test_roc_auc in results.csv first.")
        return

    plot_main_results(df, args.out)
    plot_shot_curves(df, args.out)
    plot_heatmap(df, args.out, metric="test_roc_auc")
    plot_heatmap(df, args.out, metric="test_acc")
    plot_heatmap_flat(df, args.out, metric="test_roc_auc")
    plot_heatmap_flat(df, args.out, metric="test_acc")
    plot_cross_task_delta(df, args.out)

    print(f"\nDone. Plots saved to: {args.out}/")


if __name__ == "__main__":
    main()
