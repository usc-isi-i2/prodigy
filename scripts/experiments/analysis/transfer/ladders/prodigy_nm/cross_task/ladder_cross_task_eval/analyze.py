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
            "accuracy": row.accuracy,
        })
    for row in nm2500.itertuples(index=False):
        rows.append({
            "task": "neighbor_matching", "step": 2500, "training_seed": row.seed,
            "order": row.order, "rung": row.rung, "target": row.target,
            "accuracy": row.score,
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
                "accuracy": row.accuracy,
            })
    for row in downstream2500.itertuples(index=False):
        rows.append({
            "task": "classification", "step": 2500,
            "training_seed": row.training_seed, "order": row.order,
            "rung": row.rung, "target": row.target, "accuracy": row.accuracy,
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


def summarize(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_seed = (
        cells.groupby(["task", "step", "training_seed", "order", "rung"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), n_targets=("target", "nunique"))
    )
    summary = (
        by_seed.groupby(["task", "step", "order", "rung"], as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            sd_accuracy=("accuracy", "std"),
            n_training_seeds=("training_seed", "nunique"),
        )
    )
    mix_rows = []
    for (task, step, order), group in summary.groupby(["task", "step", "order"]):
        group = group.sort_values("rung")
        best = group.loc[group["mean_accuracy"].idxmax()]
        final = group[group["rung"] == 9].iloc[0]
        gap = float(final["mean_accuracy"] - best["mean_accuracy"])
        mix_rows.append({
            "task": task,
            "step": step,
            "order": order,
            "best_rung": int(best["rung"]),
            "best_accuracy": float(best["mean_accuracy"]),
            "final_accuracy": float(final["mean_accuracy"]),
            "final_minus_best": gap,
            "mix_is_exact_max": bool(np.isclose(gap, 0.0, atol=1e-12)),
            "mix_within_1pp_of_max": bool(gap >= -0.01),
        })
    mix = pd.DataFrame(mix_rows)
    return by_seed, summary, mix


def budget_correlations(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, group in summary.groupby("task"):
        wide = group.pivot(index=["order", "rung"], columns="step", values="mean_accuracy")
        rows.append({
            "task": task,
            "n_logical_points": len(wide),
            "pearson": wide[100].corr(wide[2500], method="pearson"),
            "spearman": wide[100].corr(wide[2500], method="spearman"),
        })
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, output: Path) -> None:
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.7), sharex=True, sharey="row")
    task_rows = [
        ("neighbor_matching", "Native neighbor matching", 1 / 30),
        ("classification", "Downstream classification", 0.5),
    ]
    for row_index, (task, task_label, chance) in enumerate(task_rows):
        task_data = summary[summary["task"] == task]
        for column_index, order in enumerate(ORDERS):
            ax = axes[row_index, column_index]
            ax.axhline(chance, color=CHANCE, linewidth=1.0, linestyle=(0, (3, 3)), zorder=0)
            for step, color in ((100, STEP100), (2500, STEP2500)):
                group = task_data[(task_data["step"] == step) & (task_data["order"] == order)].sort_values("rung")
                x = group["rung"].to_numpy(dtype=float)
                y = group["mean_accuracy"].to_numpy(dtype=float)
                ax.plot(x, y, color=color, linewidth=2.1, marker="o", markersize=4.2, zorder=3)
                if step == 2500:
                    sd = group["sd_accuracy"].fillna(0).to_numpy(dtype=float)
                    ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.13, linewidth=0, zorder=1)
                best = int(group.loc[group["mean_accuracy"].idxmax(), "rung"])
                best_y = float(group.loc[group["rung"] == best, "mean_accuracy"].iloc[0])
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
                ax.set_ylabel(f"{task_label}\nmean accuracy across targets")
            if row_index == 1:
                ax.set_xlabel("Number of source graphs (ladder rung)")

    axes[0, 0].set_ylim(0.025, 0.315)
    axes[1, 0].set_ylim(0.485, 0.77)
    legend = [
        Line2D([0], [0], color=STEP100, marker="o", linewidth=2.1, label="100 steps · seed 0"),
        Line2D([0], [0], color=STEP2500, marker="o", linewidth=2.1, label="2,500 steps · 3-seed mean ± SD"),
        Line2D([0], [0], color=CHANCE, linestyle=(0, (3, 3)), linewidth=1.0, label="Chance"),
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor=INK,
               linestyle="None", label="Best rung at that budget"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=4, frameon=False)
    fig.suptitle(
        "Training budget changes the source-composition ladder",
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


def main() -> int:
    cells = load_cells()
    by_seed, summary, mix = summarize(cells)
    correlations = budget_correlations(summary)
    by_seed.to_csv(DATA / "curve_by_seed.csv", index=False)
    summary.to_csv(DATA / "curve_summary.csv", index=False)
    mix.to_csv(DATA / "mix_is_max.csv", index=False)
    correlations.to_csv(DATA / "budget_rank_correlations.csv", index=False)
    plot(summary, FIGURES / "budget_task_ladders.png")
    print(mix.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
