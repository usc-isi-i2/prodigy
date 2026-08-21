from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["roc_auc_ovr_macro", "f1_macro", "accuracy"]
METRIC_LABELS = {
    "roc_auc_ovr_macro": "Macro OVR AUC",
    "f1_macro": "Macro F1",
    "accuracy": "Accuracy",
}
TARGET_LABELS = {
    "covid_political": "COVID political",
    "ukr_rus_suspended": "UKR/RUS suspended",
    "election2020": "Election 2020",
    "twibot20": "TwiBot-20",
    "facebook_page_reference": "Facebook pages",
    "cora": "Cora",
    "pubmed": "PubMed",
}
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, output_root: Path, stem: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output_root / f"{stem}.{suffix}", facecolor="white")
    plt.close(fig)


def plot_adaptation_efficiency(data: pd.DataFrame, output_root: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.25), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        subset = data[data.metric == metric].sort_values("mixture_size")
        ax.plot(
            subset.mixture_size,
            subset.mean_log_step_aulc,
            color=COLORS[0],
            marker="o",
            linewidth=2,
        )
        best = subset.loc[subset.mean_log_step_aulc.idxmax()]
        ax.scatter([best.mixture_size], [best.mean_log_step_aulc], s=65, facecolors="white", edgecolors=COLORS[4], linewidths=2, zorder=4)
        ax.annotate(
            f"best: {int(best.mixture_size)} sources",
            (best.mixture_size, best.mean_log_step_aulc),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            color=COLORS[4],
            fontsize=8,
        )
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("Pretraining mixture size (graphs)")
        ax.set_ylabel("Adaptation efficiency")
        ax.set_xticks(range(1, 7))
    fig.suptitle("Held-out adaptation efficiency is not monotonic in mixture size", fontsize=12, fontweight="bold")
    save(fig, output_root, "adaptation-efficiency-by-mixture-size")


def plot_checkpoint_trajectories(data: pd.DataFrame, output_root: Path) -> None:
    means = data.groupby(["mixture_size", "checkpoint_step"], as_index=False)[METRICS].mean()
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        for size, color in zip(range(1, 7), COLORS):
            subset = means[means.mixture_size == size].sort_values("checkpoint_step")
            ax.plot(subset.checkpoint_step, subset[metric], marker="o", linewidth=1.7, markersize=4, color=color, label=f"{size}")
        ax.set_xscale("log")
        ax.set_xticks([100, 300, 900, 2500], labels=["100", "300", "900", "2.5k"])
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("SSL updates")
        ax.set_ylabel("Mean held-out score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Sources", ncol=6, loc="outside lower center", frameon=False)
    fig.suptitle("Downstream checkpoint trajectories across mixture sizes", fontsize=12, fontweight="bold")
    save(fig, output_root, "checkpoint-trajectories-by-mixture-size")


def heatmap(ax: plt.Axes, matrix: pd.DataFrame, title: str, fmt: str = ".3f"):
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=np.nanmin(values), vmax=np.nanmax(values))
    midpoint = (np.nanmin(values) + np.nanmax(values)) / 2
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] < midpoint else "black"
            ax.text(j, i, format(values[i, j], fmt), ha="center", va="center", fontsize=6.8, color=color)
    ax.set_title(title)
    return image


def plot_target_mixture_heatmap(data: pd.DataFrame, output_root: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.5), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        subset = data[data.metric == metric]
        matrix = subset.pivot(index="target", columns="mixture_size", values="log_step_aulc")
        matrix = matrix.reindex(TARGET_LABELS)
        image = heatmap(ax, matrix, METRIC_LABELS[metric])
        ax.set_xticks(range(6), labels=range(1, 7))
        ax.set_xlabel("Mixture size")
        ax.set_yticks(range(7), labels=[TARGET_LABELS[target] for target in matrix.index] if ax is axes[0] else [])
        fig.colorbar(image, ax=ax, shrink=0.72, pad=0.02)
    fig.suptitle("Best mixture size depends on the held-out target", fontsize=12, fontweight="bold")
    save(fig, output_root, "target-by-mixture-size-heatmap")


def short_run_label(run: str) -> str:
    if run.startswith("specialist_"):
        graph = run.removeprefix("specialist_").removesuffix("_s0")
        return f"S: {TARGET_LABELS[graph]}"
    graph = run.removeprefix("loo_").removesuffix("_s0")
    return f"LOO: {TARGET_LABELS[graph]}"


def plot_transfer_matrix(data: pd.DataFrame, output_root: Path) -> None:
    run_order = list(dict.fromkeys(data.run_id))
    target_order = list(TARGET_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 7.0), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        matrix = data.pivot(index="run_id", columns="target", values=metric).reindex(index=run_order, columns=target_order)
        image = heatmap(ax, matrix, METRIC_LABELS[metric])
        ax.set_xticks(range(7), labels=[TARGET_LABELS[target] for target in target_order], rotation=55, ha="right")
        ax.set_yticks(range(14), labels=[short_run_label(run) for run in run_order] if ax is axes[0] else [])
        ax.set_xlabel("Evaluation graph")
        fig.colorbar(image, ax=ax, shrink=0.5, pad=0.02)
    fig.suptitle("Full GraphSAGE transfer matrix at 2,500 SSL updates", fontsize=12, fontweight="bold")
    save(fig, output_root, "full-transfer-matrix-step2500")


def plot_seed_contrasts(data: pd.DataFrame, output_root: Path) -> None:
    subset = data[data.checkpoint_step == 2500].copy()
    target_order = list(TARGET_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.5), constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        metric_rows = subset[subset.metric == metric].set_index("target").reindex(target_order)
        y = np.arange(len(target_order))
        values = metric_rows.mean_loo_minus_specialist.to_numpy()
        errors = metric_rows.sample_std.to_numpy()
        colors = [COLORS[2] if value > 0 else COLORS[4] for value in values]
        ax.axvline(0, color="#555555", linewidth=1)
        ax.errorbar(values, y, xerr=errors, fmt="none", ecolor="#777777", elinewidth=1, capsize=2, zorder=1)
        ax.scatter(values, y, c=colors, s=34, zorder=2)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("LOO − target specialist")
        ax.set_yticks(y, labels=[TARGET_LABELS[target] for target in target_order] if ax is axes[0] else [])
        ax.invert_yaxis()
    fig.suptitle("Three-seed transfer gap at 2,500 SSL updates", fontsize=12, fontweight="bold")
    save(fig, output_root, "three-seed-loo-vs-specialist")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    analysis = root / "results/analysis"
    output = root / "results/figures"
    style()
    plot_adaptation_efficiency(pd.read_csv(analysis / "adaptation_by_size.csv"), output)
    plot_checkpoint_trajectories(pd.read_csv(analysis / "heldout_ladder.csv"), output)
    plot_target_mixture_heatmap(pd.read_csv(analysis / "adaptation_by_target.csv"), output)
    plot_transfer_matrix(pd.read_csv(analysis / "matrix_step2500.csv"), output)
    plot_seed_contrasts(pd.read_csv(analysis / "primary_contrasts.csv"), output)
    print(f"wrote 5 figures as PNG and PDF to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
