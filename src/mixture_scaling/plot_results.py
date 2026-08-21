from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
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
GRAPH_COLORS = {
    graph: color
    for graph, color in zip(TARGET_LABELS, COLORS + ["#6F4E7C"])
}


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
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        fig.savefig(output_root / f"{stem}.{suffix}", facecolor="white", metadata=metadata)
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


def primary_target(run_id: str) -> str:
    prefix = "specialist_" if run_id.startswith("specialist_") else "loo_"
    return run_id.removeprefix(prefix).rsplit("_s", 1)[0]


def target_axes(title: str) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, grid = plt.subplots(2, 4, figsize=(12.0, 6.4), constrained_layout=True)
    axes = list(grid.flat)
    axes[-1].axis("off")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    return fig, axes[:-1]


def format_update_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value / 1000:g}"))
    ax.set_xlabel("SSL updates (thousands)")


def plot_primary_seed0_losses(data: pd.DataFrame, output_root: Path) -> None:
    subset = data[data.experiment == "primary_s0"].copy()
    subset["target"] = subset.run_id.map(primary_target)
    fig, axes = target_axes("Seed-0 SSL convergence: target specialist versus held-out mixture")
    family_style = {
        "specialist": (COLORS[0], "Specialist"),
        "leave_one_out": (COLORS[4], "Leave-one-out"),
    }
    for ax, target in zip(axes, TARGET_LABELS):
        panel = subset[subset.target == target]
        for family, (color, label) in family_style.items():
            run = panel[panel.kind == family].sort_values("step")
            ax.plot(run.step, run.validation_loss, color=color, linewidth=2, label=f"{label} validation")
            ax.plot(run.step, run.train_loss, color=color, linewidth=1, linestyle="--", alpha=0.55, label=f"{label} train")
            best_step = int(run.best_step.iloc[0])
            best = run.loc[(run.step - best_step).abs().idxmin()]
            ax.scatter(best.step, best.validation_loss, marker="*", s=65, color=color, edgecolor="white", linewidth=0.5, zorder=4)
        ax.set_title(TARGET_LABELS[target])
        format_update_axis(ax)
        ax.set_ylabel("Binary SSL loss")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="outside lower center", frameon=False)
    save(fig, output_root, "ssl-loss-primary-seed0")


def plot_ladder_losses(data: pd.DataFrame, output_root: Path) -> None:
    subset = data[data.experiment == "ladder_s0"].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    for size, color in zip(range(2, 6), COLORS[1:5]):
        runs = subset[subset.mixture_size == size]
        first = True
        for _, run in runs.groupby("run_id"):
            run = run.sort_values("step")
            ax.plot(run.step, run.validation_loss, color=color, alpha=0.32, linewidth=1, label=f"{size} sources" if first else None)
            ax.scatter(run.step.iloc[-1], run.validation_loss.iloc[-1], color=color, alpha=0.55, s=11)
            first = False
        # Every run reaches 2,500 updates. Past that point, a mean over only
        # runs that have not early-stopped would be survivor-biased.
        mean = runs[runs.step <= 2500].groupby("step", as_index=False).validation_loss.mean()
        ax.plot(mean.step, mean.validation_loss, color=color, linewidth=2.3)
    ax.set_title("Intermediate mixtures reach SSL convergence by 2.5k–4.25k updates", fontweight="bold")
    format_update_axis(ax)
    ax.set_ylabel("Validation loss")
    ax.legend(frameon=False, ncol=2)
    save(fig, output_root, "ssl-loss-intermediate-mixtures")


def plot_three_seed_losses(data: pd.DataFrame, output_root: Path) -> None:
    subset = data[data.experiment.isin(["primary_s0", "primary_s1_s2"])].copy()
    subset["target"] = subset.run_id.map(primary_target)
    fig, axes = target_axes("SSL validation-loss robustness across three seeds")
    family_style = {"specialist": (COLORS[0], "Specialist"), "leave_one_out": (COLORS[4], "Leave-one-out")}
    seed_style = {0: "-", 1: "--", 2: ":"}
    for ax, target in zip(axes, TARGET_LABELS):
        panel = subset[subset.target == target]
        for family, (color, label) in family_style.items():
            for seed, linestyle in seed_style.items():
                run = panel[(panel.kind == family) & (panel.seed == seed)].sort_values("step")
                ax.plot(run.step, run.validation_loss, color=color, linestyle=linestyle, linewidth=1.55, alpha=0.88, label=f"{label}, seed {seed}")
        ax.set_title(TARGET_LABELS[target])
        format_update_axis(ax)
        ax.set_ylabel("Validation loss")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="outside lower center", frameon=False)
    save(fig, output_root, "ssl-loss-primary-three-seeds")


def plot_per_source_losses(data: pd.DataFrame, output_root: Path) -> None:
    subset = data[(data.experiment == "primary_s0") & (data.kind == "leave_one_out")].copy()
    subset["target"] = subset.run_id.map(primary_target)
    fig, axes = target_axes("Per-source validation loss within seed-0 held-out mixtures")
    for ax, target in zip(axes, TARGET_LABELS):
        panel = subset[subset.target == target]
        for source, run in panel.groupby("validation_source"):
            run = run.sort_values("step")
            ax.plot(run.step, run.validation_loss, color=GRAPH_COLORS[source], linewidth=1.5, label=TARGET_LABELS[source])
        ax.set_title(f"Held out: {TARGET_LABELS[target]}")
        format_update_axis(ax)
        ax.set_ylabel("Validation loss")
    handles_by_label = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles_by_label[label] = handle
    fig.legend(handles_by_label.values(), handles_by_label, ncol=4, loc="outside lower center", frameon=False)
    save(fig, output_root, "ssl-loss-per-source-seed0-loo")


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
    loss_history = pd.read_csv(root / "results/loss_curves/loss_history.csv")
    per_source_history = pd.read_csv(root / "results/loss_curves/per_source_validation_loss.csv")
    plot_primary_seed0_losses(loss_history, output)
    plot_ladder_losses(loss_history, output)
    plot_three_seed_losses(loss_history, output)
    plot_per_source_losses(per_source_history, output)
    print(f"wrote 9 figures as PNG and PDF to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
