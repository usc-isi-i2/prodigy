#!/usr/bin/env python3
"""Compare every native-GraphCL ladder cell with its best available specialist."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
LADDER_ROOT = HERE / "data" / "samgpt_graphcl_9x3_carc_v100"
MATRIX_ROOT = HERE / "data" / "samgpt_graphcl_specialist_matrix_tucker_h100"
OUT_ROOT = HERE / "data" / "samgpt_graphcl_max_rule"
FIG_ROOT = HERE / "figures"

METRIC_RULES = {
    "loss": "min",
    "accuracy": "max",
    "probability_margin": "max",
}
COLORS = {"A": "#2a78d6", "B": "#d85a30", "C": "#6f4b8b"}
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
    }
)


def validate(ladder: pd.DataFrame, matrix: pd.DataFrame) -> None:
    if len(ladder) != 243 or ladder[["order", "rung", "target"]].duplicated().any():
        raise ValueError("Ladder must contain 243 unique order/rung/target cells")
    if len(matrix) != 81 or matrix[["train_source", "target"]].duplicated().any():
        raise ValueError("Specialist matrix must contain 81 unique source/target cells")
    if set(ladder["target"]) != set(matrix["target"]):
        raise ValueError("Ladder and specialist targets differ")
    sizes = ladder.groupby(["order", "rung"]).size()
    if set(sizes) != {9} or len(sizes) != 27:
        raise ValueError("Each ladder row must contain all nine targets")
    if set(ladder["eval_seed"]) != set(matrix["eval_seed"]):
        raise ValueError("Ladder and specialist matrices use different evaluation seeds")


def build_cells(ladder: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    lookup = matrix.set_index(["train_source", "target"])
    rows = []
    for raw in ladder.itertuples(index=False):
        sources = raw.sources.split(",")
        row = {
            "order": raw.order,
            "rung": int(raw.rung),
            "added": raw.added,
            "target": raw.target,
            "sources": raw.sources,
            "source_count": int(raw.source_count),
            "in_training": bool(raw.in_training),
        }
        for metric, rule in METRIC_RULES.items():
            candidates = {
                source: float(lookup.loc[(source, raw.target), metric])
                for source in sources
            }
            winner = (min if rule == "min" else max)(candidates, key=candidates.get)
            predicted = candidates[winner]
            mean_predicted = float(np.mean(list(candidates.values())))
            observed = float(getattr(raw, metric))
            row.update(
                {
                    f"observed_{metric}": observed,
                    f"max_rule_{metric}": predicted,
                    f"mean_rule_{metric}": mean_predicted,
                    f"residual_{metric}": observed - predicted,
                    f"abs_residual_{metric}": abs(observed - predicted),
                    f"mean_rule_residual_{metric}": observed - mean_predicted,
                    f"mean_rule_abs_residual_{metric}": abs(observed - mean_predicted),
                    f"winning_specialist_{metric}": winner,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def stats(values: pd.DataFrame, metric: str) -> dict[str, float | int]:
    residual = values[f"residual_{metric}"].to_numpy(dtype=float)
    observed = values[f"observed_{metric}"].to_numpy(dtype=float)
    predicted = values[f"max_rule_{metric}"].to_numpy(dtype=float)
    return {
        "cells": len(values),
        "signed_mean_residual": float(np.mean(residual)),
        "mae": float(np.mean(np.abs(residual))),
        "median_absolute_error": float(np.median(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "max_absolute_error": float(np.max(np.abs(residual))),
        "pearson_r": float(np.corrcoef(observed, predicted)[0, 1]),
        "fraction_within_0_001": float(np.mean(np.abs(residual) <= 0.001)),
    }


def build_summary(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRIC_RULES:
        rows.append({"metric": metric, "scope": "overall", "order": "all", "rung": "all", **stats(cells, metric)})
        for order, values in cells.groupby("order"):
            rows.append({"metric": metric, "scope": "order", "order": order, "rung": "all", **stats(values, metric)})
        for (order, rung), values in cells.groupby(["order", "rung"]):
            rows.append({"metric": metric, "scope": "order_rung", "order": order, "rung": rung, **stats(values, metric)})
    return pd.DataFrame(rows)


def build_rule_comparison_summary(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRIC_RULES:
        observed = cells[f"observed_{metric}"].to_numpy(dtype=float)
        for rule in ("max", "mean"):
            predicted = cells[f"{rule}_rule_{metric}"].to_numpy(dtype=float)
            residual = observed - predicted
            rows.append(
                {
                    "metric": metric,
                    "rule": rule,
                    "cells": len(cells),
                    "mae": float(np.mean(np.abs(residual))),
                    "median_absolute_error": float(np.median(np.abs(residual))),
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "pearson_r": float(np.corrcoef(observed, predicted)[0, 1]),
                    "fraction_within_0_001": float(np.mean(np.abs(residual) <= 0.001)),
                }
            )
    return pd.DataFrame(rows)


def chrome(ax: plt.Axes) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.grid(color=GRID, lw=0.75, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=8.7)


def plot(cells: pd.DataFrame, overall: dict[str, float | int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 5.25), dpi=200)
    ax = axes[0]
    for order, values in cells.groupby("order"):
        ax.scatter(
            values["max_rule_loss"],
            values["observed_loss"],
            s=25,
            color=COLORS[order],
            alpha=0.68,
            edgecolor="white",
            linewidth=0.35,
            label=f"order {order}",
            zorder=3,
        )
    limits = [1e-6, 0.8]
    ax.plot(limits, limits, color=INK, lw=1.3, ls=(0, (4, 2)), zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("max-rule prediction: minimum constituent-specialist loss", fontsize=9.5)
    ax.set_ylabel("observed mixture GraphCL loss", fontsize=9.5)
    ax.set_title("243 target-level comparisons", loc="left", fontsize=11, fontweight="bold")
    ax.text(
        0.03,
        0.96,
        f"Pearson r = {overall['pearson_r']:.3f}\n"
        f"median |error| = {overall['median_absolute_error']:.6f}\n"
        f"within .001 = {overall['fraction_within_0_001']:.1%}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.7,
        color=INK,
    )
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    chrome(ax)

    ax = axes[1]
    rungs = np.arange(1, 10)
    for order in ("A", "B", "C"):
        means = cells[cells["order"].eq(order)].groupby("rung")[
            ["observed_loss", "max_rule_loss"]
        ].mean()
        color = COLORS[order]
        ax.plot(rungs, means["observed_loss"], color=color, lw=2.2, marker="o", ms=5, zorder=4)
        ax.plot(rungs, means["max_rule_loss"], color=color, lw=1.5, ls=(0, (4, 2)), marker="s", ms=4, alpha=0.82, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(rungs)
    ax.set_xlabel("cumulative mixture rung", fontsize=9.5)
    ax.set_ylabel("mean loss over the fixed nine targets", fontsize=9.5)
    ax.set_title("Observed vs predicted mean curves", loc="left", fontsize=11, fontweight="bold")
    handles = [
        Line2D([0], [0], color=COLORS[order], lw=2.2, label=f"order {order}")
        for order in ("A", "B", "C")
    ] + [
        Line2D([0], [0], color=INK, lw=2.2, marker="o", label="observed mixture"),
        Line2D([0], [0], color=INK, lw=1.5, ls=(0, (4, 2)), marker="s", label="max-rule prediction"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.2, ncol=2, loc="upper right")
    chrome(ax)

    fig.suptitle(
        "SAMGPT native objective: the constituent-specialist max rule is approximate",
        x=0.06,
        ha="left",
        fontsize=13.2,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.93,
        "loss is lower-better, so the max-performance rule is the minimum specialist loss  ·  "
        "ladder evaluated on CARC V100; specialist matrix on Tucker H100  ·  same fixed unseen views",
        ha="left",
        fontsize=8.7,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=2.4)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = FIG_ROOT / f"samgpt_graphcl_max_rule.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=220)
        print("wrote", path)
    plt.close(fig)


def plot_bounded_metric(cells: pd.DataFrame, comparison: pd.DataFrame, metric: str) -> None:
    if metric not in {"accuracy", "probability_margin"}:
        raise ValueError(f"Unsupported bounded metric: {metric}")
    label = "accuracy" if metric == "accuracy" else "probability margin"
    observed = cells[f"observed_{metric}"].to_numpy(dtype=float)
    max_predicted = cells[f"max_rule_{metric}"].to_numpy(dtype=float)
    mean_predicted = cells[f"mean_rule_{metric}"].to_numpy(dtype=float)
    stats_by_rule = comparison[comparison["metric"].eq(metric)].set_index("rule")

    fig, axes = plt.subplots(1, 2, figsize=(12.7, 5.25), dpi=200)
    ax = axes[0]
    ax.scatter(
        mean_predicted,
        observed,
        s=23,
        color="#a6a39b",
        alpha=0.52,
        marker="x",
        linewidth=0.8,
        label="mean rule",
        zorder=2,
    )
    ax.scatter(
        max_predicted,
        observed,
        s=24,
        color="#2a78d6",
        alpha=0.68,
        edgecolor="white",
        linewidth=0.35,
        label="max rule",
        zorder=3,
    )
    lower = min(observed.min(), max_predicted.min(), mean_predicted.min())
    upper = max(observed.max(), max_predicted.max(), mean_predicted.max())
    pad = max((upper - lower) * 0.07, 0.005)
    lower_bound = 0.0 if metric == "accuracy" else -1.0
    limits = [max(lower_bound, lower - pad), upper + pad]
    ax.plot(limits, limits, color=INK, lw=1.3, ls=(0, (4, 2)), zorder=1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("rule prediction", fontsize=9.5)
    ax.set_ylabel(f"observed mixture {label}", fontsize=9.5)
    ax.set_title("243 target-level comparisons", loc="left", fontsize=11, fontweight="bold")
    ax.text(
        0.97,
        0.06,
        f"max MAE = {stats_by_rule.loc['max', 'mae']:.5f}\n"
        f"mean MAE = {stats_by_rule.loc['mean', 'mae']:.5f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.7,
        color=INK,
    )
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    chrome(ax)

    ax = axes[1]
    rung_errors = cells.groupby("rung").agg(
        max_mae=(f"abs_residual_{metric}", "mean"),
        mean_mae=(f"mean_rule_abs_residual_{metric}", "mean"),
    )
    rungs = rung_errors.index.to_numpy(dtype=int)
    ax.plot(
        rungs,
        rung_errors["mean_mae"],
        color="#a6a39b",
        lw=1.7,
        marker="x",
        ms=6,
        label="mean rule",
        zorder=2,
    )
    ax.plot(
        rungs,
        rung_errors["max_mae"],
        color="#2a78d6",
        lw=2.2,
        marker="o",
        ms=5,
        markeredgecolor="white",
        markeredgewidth=0.7,
        label="max rule",
        zorder=3,
    )
    ax.set_xticks(rungs)
    ax.set_xlabel("cumulative mixture rung", fontsize=9.5)
    ax.set_ylabel(f"mean absolute {label} error", fontsize=9.5)
    ax.set_title("Prediction error by rung", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    chrome(ax)

    improvement = 1.0 - (
        stats_by_rule.loc["max", "mae"] / stats_by_rule.loc["mean", "mae"]
    )
    fig.suptitle(
        f"SAMGPT native objective: max versus mean rule on {label}",
        x=0.06,
        ha="left",
        fontsize=13.2,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.93,
        f"{label} is bounded  ·  max-rule MAE is {improvement:.1%} lower than mean-rule MAE  ·  "
        "same fixed unseen corruption/edge-drop views",
        ha="left",
        fontsize=8.7,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=2.4)
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = FIG_ROOT / f"samgpt_graphcl_max_vs_mean_{metric}.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=220)
        print("wrote", path)
    plt.close(fig)


def main() -> None:
    ladder = pd.read_csv(LADDER_ROOT / "metrics_long.csv")
    matrix = pd.read_csv(MATRIX_ROOT / "metrics_long.csv")
    validate(ladder, matrix)
    cells = build_cells(ladder, matrix)
    summary = build_summary(cells)
    comparison = build_rule_comparison_summary(cells)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cells.to_csv(OUT_ROOT / "cells.csv", index=False)
    summary.to_csv(OUT_ROOT / "summary.csv", index=False)
    comparison.to_csv(OUT_ROOT / "rule_comparison_summary.csv", index=False)
    global_rows = summary[summary["scope"].eq("overall")].set_index("metric")
    global_summary = {
        metric: {
            key: (int(value) if key == "cells" else float(value))
            for key, value in global_rows.loc[metric].items()
            if key not in {"scope", "order", "rung"}
        }
        for metric in METRIC_RULES
    }
    manifest = {
        "ladder_data": str(LADDER_ROOT / "metrics_long.csv"),
        "specialist_matrix_data": str(MATRIX_ROOT / "metrics_long.csv"),
        "comparison_cells": len(cells),
        "rule": {
            "loss": "minimum among specialists present in the rung",
            "accuracy": "maximum among specialists present in the rung",
            "probability_margin": "maximum among specialists present in the rung",
        },
        "mean_rule": "arithmetic mean among specialists present in the rung",
        "rule_comparison_summary": str(OUT_ROOT / "rule_comparison_summary.csv"),
        "global_summary": global_summary,
    }
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    plot(cells, global_summary["loss"])
    plot_bounded_metric(cells, comparison, "probability_margin")
    plot_bounded_metric(cells, comparison, "accuracy")
    print(json.dumps(global_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
