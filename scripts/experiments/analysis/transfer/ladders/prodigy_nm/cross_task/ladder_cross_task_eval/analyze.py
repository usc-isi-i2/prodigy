#!/usr/bin/env python3
"""Assemble and plot the crossed PRODIGY ladder task/budget grid."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from scripts.experiments.setup.final_core.core_plan import ORDERS, build_models


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"
REPO_ROOT = next(
    parent for parent in HERE.parents
    if (parent / "experiments").is_dir() and (parent / "scripts").is_dir()
)
NM2500 = (
    REPO_ROOT
    / "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/data"
    / "prodigy_final_core/fixed_test/summary/ladder_results_alias_expanded.tsv"
)
NM2500_FINGERPRINTS = (
    REPO_ROOT
    / "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/data"
    / "prodigy_final_core/fixed_test/summary/episode_fingerprints.tsv"
)
NM2500_LOGGED_METRICS = (
    REPO_ROOT
    / "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/data"
    / "prodigy_final_core/log_recovered_metrics/physical_metrics.tsv"
)
NM40000_HISTORICAL = (
    REPO_ROOT
    / "scripts/experiments/analysis/transfer/ladders/prodigy_nm/robustness"
    / "nm_ladder_order_robustness/data/nm_ladder_order_robustness_long.csv"
)
SATURATION_H1 = (
    REPO_ROOT
    / "scripts/experiments/analysis/transfer/ablations/prodigy_nm/saturation"
    / "pretrain_saturation/data/pretrain_saturation_long.csv"
)
SATURATION_H2 = (
    REPO_ROOT
    / "scripts/experiments/analysis/transfer/ablations/prodigy_nm/saturation"
    / "pretrain_saturation_nhop2/data/pretrain_saturation_nhop2_long.csv"
)
DOWNSTREAM100 = (
    REPO_ROOT
    / "scripts/experiments/analysis/transfer/matrices/cross_architecture/icl_arch_matrix"
    / "data/raw_aggregate/prodigy.jsonl"
)

INK = "#171717"
MUTED = "#716f69"
GRID = "#deddd7"
STEP100 = "#d36b3f"
STEP2500 = "#2878b8"
STEP40000 = "#43835c"
CHANCE = "#8c8982"
ORDER_LABELS = {
    "A": "UKR/RUS → … → Facebook pages",
    "B": "UKR suspended → … → TwiBot-20",
    "C": "TwiBot-20 → … → UKR suspended",
}


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def model_for_ladder(order: str, rung: int) -> str:
    wanted = frozenset(ORDERS[order][:rung])
    matches = [model.model_id for model in build_models() if frozenset(model.sources) == wanted]
    if len(matches) != 1:
        raise AssertionError(f"expected one physical model for {order}/{rung}")
    return matches[0]


def _validate_fingerprints(manifest: dict, downstream100: pd.DataFrame) -> None:
    published_nm = pd.read_csv(NM2500_FINGERPRINTS, sep="\t").set_index("target")
    for target, values in manifest["nm_episode_fingerprints"].items():
        for field in ("episode_plan_fingerprint", "observed_episode_fingerprint"):
            if values[field] != published_nm.loc[target, field]:
                raise ValueError(f"NM fingerprint mismatch for {target}/{field}")
    published_downstream = downstream100.groupby("dataset")["episode_fingerprint"].nunique()
    if not (published_downstream == 1).all():
        raise ValueError("step-100 downstream fingerprints drift within the published file")
    expected = downstream100.groupby("dataset")["episode_fingerprint"].first().to_dict()
    if manifest["downstream_episode_fingerprints"] != expected:
        raise ValueError("step-2,500 downstream streams differ from the step-100 streams")


def load_cells() -> pd.DataFrame:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    if manifest["nm_physical_cells"] != 225 or manifest["downstream_physical_cells"] != 300:
        raise ValueError("cross-task physical grid is incomplete")

    nm100 = pd.read_csv(DATA / "nm_step100_ladder.csv")
    downstream2500 = pd.read_csv(DATA / "downstream_step2500_ladder.csv")
    nm2500 = pd.read_csv(NM2500, sep="\t")
    nm2500_metrics = pd.read_csv(NM2500_LOGGED_METRICS, sep="\t")
    if len(nm2500_metrics) != 837 or set(nm2500_metrics["printed_decimal_places"]) != {4}:
        raise ValueError("expected the complete four-decimal final-core metric recovery")
    nm2500 = nm2500.merge(
        nm2500_metrics[["seed", "model_id", "target", "roc_auc_ovr_macro_logged"]],
        on=["seed", "model_id", "target"],
        how="left",
        validate="many_to_one",
    )
    if nm2500["roc_auc_ovr_macro_logged"].isna().any():
        raise ValueError("missing log-recovered AUC for a step-2,500 NM ladder cell")
    downstream100 = pd.DataFrame(
        json.loads(line) for line in DOWNSTREAM100.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    _validate_fingerprints(manifest, downstream100)

    if len(nm100) != 243 or set(nm100["training_seed"]) != {0}:
        raise ValueError("expected 243 logical seed-0 step-100 NM cells")
    if len(nm2500) != 729 or set(nm2500["seed"]) != {0, 1, 2}:
        raise ValueError("expected 729 logical three-seed step-2,500 NM cells")
    if len(downstream2500) != 324 or set(downstream2500["training_seed"]) != {0, 1, 2}:
        raise ValueError("expected 324 logical three-seed step-2,500 downstream cells")

    rows: list[dict[str, object]] = []
    for row in nm100.itertuples(index=False):
        rows.append({
            "task": "neighbor_matching", "step": 100, "training_seed": 0,
            "order": row.order, "rung": row.rung, "target": row.target,
            "accuracy": row.accuracy, "roc_auc": row.roc_auc,
        })
    for row in nm2500.itertuples(index=False):
        rows.append({
            "task": "neighbor_matching", "step": 2500, "training_seed": row.seed,
            "order": row.order, "rung": row.rung, "target": row.target,
            "accuracy": row.score, "roc_auc": row.roc_auc_ovr_macro_logged,
        })

    aliases: dict[str, list[tuple[str, int]]] = {}
    for order in ORDERS:
        for rung in range(1, 10):
            aliases.setdefault(model_for_ladder(order, rung), []).append((order, rung))
    downstream100 = downstream100[downstream100["model_id"].isin(aliases)]
    if len(downstream100) != 100:
        raise ValueError(f"expected 100 physical step-100 downstream cells, got {len(downstream100)}")
    for row in downstream100.itertuples(index=False):
        for order, rung in aliases[row.model_id]:
            rows.append({
                "task": "classification", "step": 100, "training_seed": 0,
                "order": order, "rung": rung, "target": row.dataset,
                "accuracy": row.accuracy, "roc_auc": row.roc_auc,
            })
    for row in downstream2500.itertuples(index=False):
        rows.append({
            "task": "classification", "step": 2500,
            "training_seed": row.training_seed, "order": row.order,
            "rung": row.rung, "target": row.target, "accuracy": row.accuracy,
            "roc_auc": row.roc_auc,
        })

    cells = pd.DataFrame(rows)
    expected = {
        ("neighbor_matching", 100): 243,
        ("neighbor_matching", 2500): 729,
        ("classification", 100): 108,
        ("classification", 2500): 324,
    }
    observed = cells.groupby(["task", "step"]).size().to_dict()
    if observed != expected:
        raise ValueError(f"logical cell counts differ: {observed}")
    return cells


def summarize(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_seed = (
        cells.groupby(["task", "step", "training_seed", "order", "rung"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            roc_auc=("roc_auc", "mean"),
            n_targets=("target", "nunique"),
        )
    )
    summary = (
        by_seed.groupby(["task", "step", "order", "rung"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            sd_accuracy=("accuracy", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            sd_roc_auc=("roc_auc", "std"),
            n_training_seeds=("training_seed", "nunique"),
        )
    )
    return by_seed, summary


def mix_is_max(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    mean_column = f"mean_{metric}"
    mix_rows = []
    for (task, step, order), group in summary.groupby(["task", "step", "order"]):
        group = group.sort_values("rung")
        best = group.loc[group[mean_column].idxmax()]
        final = group[group["rung"] == 9].iloc[0]
        gap = float(final[mean_column] - best[mean_column])
        mix_rows.append({
            "task": task,
            "step": step,
            "order": order,
            "best_rung": int(best["rung"]),
            f"best_{metric}": float(best[mean_column]),
            f"final_{metric}": float(final[mean_column]),
            "final_minus_best": gap,
            "mix_is_exact_max": bool(np.isclose(gap, 0.0, atol=1e-12)),
            "mix_within_1pp_of_max": bool(gap >= -0.01),
        })
    return pd.DataFrame(mix_rows)


def budget_correlations(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for task, group in summary.groupby("task"):
        wide = group.pivot(index=["order", "rung"], columns="step", values=f"mean_{metric}")
        rows.append({
            "task": task,
            "n_logical_points": len(wide),
            "pearson": wide[100].corr(wide[2500], method="pearson"),
            "spearman": wide[100].corr(wide[2500], method="spearman"),
        })
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, output: Path, metric: str) -> None:
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.7), sharex=True, sharey="row")
    if metric == "accuracy":
        task_rows = [
            ("neighbor_matching", "Native neighbor matching", 1 / 30),
            ("classification", "Downstream classification", 0.5),
        ]
        metric_label = "mean accuracy across targets"
    elif metric == "roc_auc":
        task_rows = [
            ("neighbor_matching", "Native neighbor matching", 0.5),
            ("classification", "Downstream classification", 0.5),
        ]
        metric_label = "mean ROC-AUC across targets"
    else:
        raise ValueError(f"unsupported metric: {metric}")
    mean_column = f"mean_{metric}"
    sd_column = f"sd_{metric}"
    for row_index, (task, task_label, chance) in enumerate(task_rows):
        task_data = summary[summary["task"] == task]
        for column_index, order in enumerate(ORDERS):
            ax = axes[row_index, column_index]
            ax.axhline(chance, color=CHANCE, linewidth=1.0, linestyle=(0, (3, 3)), zorder=0)
            for step, color in ((100, STEP100), (2500, STEP2500)):
                group = task_data[(task_data["step"] == step) & (task_data["order"] == order)].sort_values("rung")
                x = group["rung"].to_numpy(dtype=float)
                y = group[mean_column].to_numpy(dtype=float)
                ax.plot(x, y, color=color, linewidth=2.1, marker="o", markersize=4.2, zorder=3)
                if step == 2500:
                    sd = group[sd_column].fillna(0).to_numpy(dtype=float)
                    ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.13, linewidth=0, zorder=1)
                best = int(group.loc[group[mean_column].idxmax(), "rung"])
                best_y = float(group.loc[group["rung"] == best, mean_column].iloc[0])
                ax.scatter(best, best_y, s=48, facecolor="white", edgecolor=color, linewidth=1.6, zorder=4)

            ax.set_title(
                f"Order {order}\n{ORDER_LABELS[order]}",
                loc="left",
                fontweight="bold",
            )
            ax.set_xticks(range(1, 10))
            ax.grid(axis="y", color=GRID, linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            if column_index == 0:
                ax.set_ylabel(f"{task_label}\n{metric_label}")
            if row_index == 1:
                ax.set_xlabel("Number of source graphs (ladder rung)")

    if metric == "accuracy":
        axes[0, 0].set_ylim(0.025, 0.315)
        axes[1, 0].set_ylim(0.485, 0.77)
    else:
        axes[0, 0].set_ylim(0.49, 0.96)
        axes[1, 0].set_ylim(0.49, 0.84)
    legend = [
        Line2D([0], [0], color=STEP100, marker="o", linewidth=2.1, label="100 steps · seed 0"),
        Line2D([0], [0], color=STEP2500, marker="o", linewidth=2.1, label="2,500 steps · 3-seed mean ± SD"),
        Line2D([0], [0], color=CHANCE, linestyle=(0, (3, 3)), linewidth=1.0, label="Chance"),
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor=INK,
               linestyle="None", label="Best rung at that budget"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=4, frameon=False)
    fig.suptitle(
        f"Training budget changes the source-composition ladder · "
        f"{'accuracy' if metric == 'accuracy' else 'ROC-AUC'}",
        x=0.055, ha="left", y=0.995, fontsize=15, fontweight="bold",
    )
    fig.text(
        0.055, 0.952,
        "Within each row, frozen evaluation streams are identical; models come from separate matched campaigns. "
        "Open markers identify the best rung; rung 9 is the shared all-nine-source model.",
        color=MUTED, fontsize=9,
    )
    fig.tight_layout(rect=(0.04, 0.03, 1, 0.89), h_pad=2.8, w_pad=1.4)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=260, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def historical_nm_budget_summary(cells: pd.DataFrame) -> pd.DataFrame:
    shared = cells[
        (cells["task"] == "neighbor_matching")
        & (cells["target"] != "facebook_page_reference")
    ]
    by_seed = (
        shared.groupby(["step", "training_seed", "order", "rung"], as_index=False)
        .agg(roc_auc=("roc_auc", "mean"), n_targets=("target", "nunique"))
    )
    current = (
        by_seed.groupby(["step", "order", "rung"], as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            sd_roc_auc=("roc_auc", "std"),
            n_training_seeds=("training_seed", "nunique"),
            n_targets=("n_targets", "first"),
        )
    )
    current["eval_protocol"] = "fixed512_static_test_on_static_train"
    current["source_order"] = "current_nine_source_order"
    current["source_set_aligned_to_current"] = True

    historical = pd.read_csv(NM40000_HISTORICAL)
    if historical.groupby("order").size().to_dict() != {"A": 64, "B": 64, "C": 64}:
        raise ValueError("expected the complete historical three-order 8x8 NM ladder")
    legacy = (
        historical.groupby(["order", "rung"], as_index=False)
        .agg(mean_roc_auc=("auc", "mean"), n_targets=("test_graph", "nunique"))
    )
    legacy["step"] = 40000
    legacy["sd_roc_auc"] = np.nan
    legacy["n_training_seeds"] = 1
    legacy["eval_protocol"] = "legacy_shared_harness"
    legacy["source_order"] = "historical_eight_source_order"
    legacy["source_set_aligned_to_current"] = legacy["order"] == "A"
    columns = [
        "step", "order", "rung", "mean_roc_auc", "sd_roc_auc",
        "n_training_seeds", "n_targets", "eval_protocol", "source_order",
        "source_set_aligned_to_current",
    ]
    combined = pd.concat([current[columns], legacy[columns]], ignore_index=True)
    return combined.sort_values(["order", "step", "rung"]).reset_index(drop=True)


def plot_historical_nm_budgets(summary: pd.DataFrame, output: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.45), sharex=True, sharey=True)
    styles = {
        100: (STEP100, "-", "100 steps · fixed512 · seed 0"),
        2500: (STEP2500, "-", "2,500 steps · fixed512 · 3-seed mean ± SD"),
        40000: (STEP40000, (0, (5, 2)), "40k · legacy harness · seed 0"),
    }
    for column_index, order in enumerate(ORDERS):
        ax = axes[column_index]
        ax.axhline(0.5, color=CHANCE, linewidth=1.0, linestyle=(0, (3, 3)), zorder=0)
        for step, (color, linestyle, _) in styles.items():
            group = summary[(summary["step"] == step) & (summary["order"] == order)].sort_values("rung")
            x = group["rung"].to_numpy(dtype=float)
            y = group["mean_roc_auc"].to_numpy(dtype=float)
            ax.plot(
                x, y, color=color, linestyle=linestyle, linewidth=2.2,
                marker="o", markersize=4.5, zorder=3,
            )
            if step == 2500:
                sd = group["sd_roc_auc"].fillna(0).to_numpy(dtype=float)
                ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.13, linewidth=0, zorder=1)
        qualifier = "source-aligned" if order == "A" else "40k uses a different source order"
        ax.set_title(f"Order {order} · {qualifier}\n{ORDER_LABELS[order]}", loc="left", fontweight="bold")
        ax.set_xticks(range(1, 10))
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.set_xlabel("Number of source graphs (ladder rung)")
    axes[0].set_ylabel("Native neighbor matching\nmean ROC-AUC across the shared 8 targets")
    axes[0].set_ylim(0.49, 0.96)
    legend = [
        Line2D([0], [0], color=color, linestyle=linestyle, marker="o", linewidth=2.2, label=label)
        for color, linestyle, label in styles.values()
    ] + [
        Line2D([0], [0], color=CHANCE, linestyle=(0, (3, 3)), linewidth=1.0, label="Chance"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.89), ncol=4, frameon=False)
    fig.suptitle(
        "PRODIGY native-NM AUC across training budgets",
        x=0.055, ha="left", y=0.995, fontsize=15, fontweight="bold",
    )
    fig.text(
        0.055, 0.94,
        "All means use the same eight targets. The 40k curve ends at rung 8 and uses the legacy evaluator; "
        "only Order A matches source sets rung-by-rung.",
        color=MUTED, fontsize=9,
    )
    fig.tight_layout(rect=(0.04, 0.03, 1, 0.82), w_pad=1.4)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=260, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def budget_phase_summary(cells: pd.DataFrame, historical_nm: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    nm = historical_nm[
        (historical_nm["order"] == "A")
        & (historical_nm["rung"] == 8)
    ]
    for row in nm.itertuples(index=False):
        rows.append({
            "task": "neighbor_matching",
            "series": "Order A rung 8 · shared 8 targets",
            "step": row.step,
            "mean_roc_auc": row.mean_roc_auc,
            "protocol": "legacy" if row.step == 40000 else "fixed512",
            "n_sources": 8,
            "n_targets": 8,
        })

    downstream = cells[(cells["task"] == "classification") & (cells["rung"] == 9)]
    downstream = (
        downstream.groupby(["step", "training_seed"], as_index=False)
        .agg(roc_auc=("roc_auc", "mean"), n_targets=("target", "nunique"))
        .groupby("step", as_index=False)
        .agg(mean_roc_auc=("roc_auc", "mean"), n_targets=("n_targets", "first"))
    )
    for row in downstream.itertuples(index=False):
        rows.append({
            "task": "classification",
            "series": "New all9 · fixed512",
            "step": row.step,
            "mean_roc_auc": row.mean_roc_auc,
            "protocol": "fixed512",
            "n_sources": 9,
            "n_targets": row.n_targets,
        })

    for path, label in ((SATURATION_H1, "Saturation all8 · one hop"), (SATURATION_H2, "Saturation all8 · two hop")):
        saturation = pd.read_csv(path)
        saturation = saturation[
            (saturation["arm"] == "all8")
            & (saturation["task"] == "classification")
            & (saturation["metric"] == "roc_auc")
            & (saturation["step"] >= 100)
        ]
        saturation = saturation.groupby("step", as_index=False).agg(
            mean_roc_auc=("value", "mean"),
            n_targets=("dataset", "nunique"),
        )
        for row in saturation.itertuples(index=False):
            rows.append({
                "task": "classification",
                "series": label,
                "step": row.step,
                "mean_roc_auc": row.mean_roc_auc,
                "protocol": "legacy_downstream",
                "n_sources": 8,
                "n_targets": row.n_targets,
            })
    return pd.DataFrame(rows).sort_values(["task", "series", "step"]).reset_index(drop=True)


def mix_regret_summary(auc_mix: pd.DataFrame) -> pd.DataFrame:
    current = auc_mix.copy()
    current["regret_auc_points"] = -100 * current["final_minus_best"]
    current["protocol"] = "fixed512"
    current["final_rung"] = 9

    historical = pd.read_csv(NM40000_HISTORICAL)
    means = historical.groupby(["order", "rung"], as_index=False).agg(mean_auc=("auc", "mean"))
    rows = []
    for order, group in means.groupby("order"):
        final = float(group.loc[group["rung"] == 8, "mean_auc"].iloc[0])
        rows.append({
            "task": "neighbor_matching",
            "step": 40000,
            "order": order,
            "best_rung": int(group.loc[group["mean_auc"].idxmax(), "rung"]),
            "regret_auc_points": 100 * (float(group["mean_auc"].max()) - final),
            "protocol": "legacy",
            "final_rung": 8,
        })
    columns = ["task", "step", "order", "best_rung", "regret_auc_points", "protocol", "final_rung"]
    return pd.concat([current[columns], pd.DataFrame(rows)[columns]], ignore_index=True).sort_values(
        ["task", "step", "order"]
    ).reset_index(drop=True)


def plot_budget_phase(summary: pd.DataFrame, output: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.65), sharey=True)
    colors = {
        "Order A rung 8 · shared 8 targets": STEP2500,
        "New all9 · fixed512": STEP2500,
        "Saturation all8 · one hop": STEP100,
        "Saturation all8 · two hop": STEP40000,
    }
    task_specs = [
        ("neighbor_matching", "Native neighbor matching"),
        ("classification", "Downstream classification"),
    ]
    for ax, (task, title) in zip(axes, task_specs):
        task_data = summary[summary["task"] == task]
        ax.axhline(0.5, color=CHANCE, linewidth=1.0, linestyle=(0, (3, 3)), zorder=0)
        ax.axvspan(100, 500, color=GRID, alpha=0.32, linewidth=0, zorder=0)
        for series, group in task_data.groupby("series"):
            group = group.sort_values("step")
            ax.plot(
                group["step"], group["mean_roc_auc"],
                color=colors[series], linewidth=2.2, marker="o", markersize=5,
                label=series, zorder=3,
            )
            legacy = group[group["protocol"].str.startswith("legacy")]
            if not legacy.empty:
                ax.scatter(
                    legacy["step"], legacy["mean_roc_auc"], s=54,
                    facecolor="white", edgecolor=colors[series], linewidth=1.8, zorder=4,
                )
        ax.set_xscale("log")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Pretraining steps (log scale)")
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    axes[0].set_ylabel("Mean ROC-AUC")
    axes[0].set_ylim(0.49, 0.95)
    fig.suptitle(
        "Most measurable transfer is acquired before the mature-budget regime",
        x=0.065, ha="left", y=0.995, fontsize=15, fontweight="bold",
    )
    fig.text(
        0.065, 0.905,
        "Shaded interval marks 100–500 steps. Open markers use a legacy evaluation campaign; lines connect contextual, not causal, comparisons.",
        color=MUTED, fontsize=9,
    )
    fig.tight_layout(rect=(0.04, 0.03, 1, 0.82), w_pad=2.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=260, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_mix_regret(summary: pd.DataFrame, output: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.45), sharey=True)
    order_colors = {"A": STEP100, "B": STEP2500, "C": STEP40000}
    task_specs = [
        ("neighbor_matching", "Native neighbor matching"),
        ("classification", "Downstream classification"),
    ]
    for ax, (task, title) in zip(axes, task_specs):
        task_data = summary[summary["task"] == task]
        for order in ORDERS:
            group = task_data[task_data["order"] == order].sort_values("step")
            ax.plot(
                group["step"], group["regret_auc_points"],
                color=order_colors[order], linewidth=2.0, marker="o", markersize=5,
                label=f"Order {order}", zorder=3,
            )
            legacy = group[group["protocol"] == "legacy"]
            if not legacy.empty:
                ax.scatter(
                    legacy["step"], legacy["regret_auc_points"], s=54,
                    facecolor="white", edgecolor=order_colors[order], linewidth=1.8, zorder=4,
                )
        ax.axhline(0, color=CHANCE, linewidth=1.0, zorder=0)
        ax.set_xscale("log")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Pretraining steps (log scale)")
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper right")
    axes[0].set_ylabel("Best earlier rung − all-source rung (AUC points)")
    axes[0].set_ylim(-0.5, 15.0)
    fig.suptitle(
        "The apparent all-source penalty is an early-training phenomenon",
        x=0.065, ha="left", y=0.995, fontsize=15, fontweight="bold",
    )
    fig.text(
        0.065, 0.905,
        "Zero means the largest mixture is best. The open 40k markers are historical eight-rung ladders under the legacy evaluator.",
        color=MUTED, fontsize=9,
    )
    fig.tight_layout(rect=(0.04, 0.03, 1, 0.82), w_pad=2.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=260, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    cells = load_cells()
    by_seed, summary = summarize(cells)
    accuracy_mix = mix_is_max(summary, "accuracy")
    auc_mix = mix_is_max(summary, "roc_auc")
    accuracy_correlations = budget_correlations(summary, "accuracy")
    auc_correlations = budget_correlations(summary, "roc_auc")
    historical_nm = historical_nm_budget_summary(cells)
    budget_phase = budget_phase_summary(cells, historical_nm)
    mix_regret = mix_regret_summary(auc_mix)
    by_seed.to_csv(DATA / "curve_by_seed.csv", index=False)
    summary.to_csv(DATA / "curve_summary.csv", index=False)
    accuracy_mix.to_csv(DATA / "mix_is_max.csv", index=False)
    auc_mix.to_csv(DATA / "mix_is_max_auc.csv", index=False)
    accuracy_correlations.to_csv(DATA / "budget_rank_correlations.csv", index=False)
    auc_correlations.to_csv(DATA / "budget_rank_correlations_auc.csv", index=False)
    historical_nm.to_csv(DATA / "nm_auc_budget_100_2500_40000.csv", index=False)
    budget_phase.to_csv(DATA / "budget_phase_transition_auc.csv", index=False)
    mix_regret.to_csv(DATA / "mix_regret_auc.csv", index=False)
    plot(summary, FIGURES / "budget_task_ladders.png", "accuracy")
    plot(summary, FIGURES / "budget_task_ladders_auc.png", "roc_auc")
    plot_historical_nm_budgets(
        historical_nm,
        FIGURES / "nm_auc_budget_100_2500_40000.png",
    )
    plot_budget_phase(budget_phase, FIGURES / "budget_phase_transition_auc.png")
    plot_mix_regret(mix_regret, FIGURES / "mix_regret_auc.png")
    print(auc_mix.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
