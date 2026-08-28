#!/usr/bin/env python3
"""Plot the primary cross-target selection result against the fixed-100 legacy endpoint."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
ORDER = ["PRODIGY", "VISION", "SAMGPT", "GraphSAGE", "Raw logistic", "Raw MLP"]
COLORS = dict(zip(ORDER, plt.get_cmap("tab10").colors[: len(ORDER)]))
MARKERS = dict(zip(ORDER, ("o", "s", "^", "D", "P", "X")))
BUDGETS = np.asarray([1, 10, 100])
TARGET_ORDER = ["covid_political", "election2020", "ukr_rus_suspended", "twibot20"]
TARGET_LABELS = {
    "covid_political": "COVID political",
    "election2020": "Election 2020",
    "ukr_rus_suspended": "Ukraine suspended",
    "twibot20": "TwiBot-20",
}


def extended_cross_target_selected() -> pd.DataFrame:
    """Apply the registered leave-one-target-out update selection to the extended grid."""
    cells = pd.read_csv(DATA / "adaptation_cells_extended.csv")
    cells["family"] = cells.model_id.map(
        lambda model: next(
            family
            for prefix, family in (
                ("prodigy_", "PRODIGY"),
                ("vision_", "VISION"),
                ("samgpt_", "SAMGPT"),
                ("graphsage_", "GraphSAGE"),
                ("raw_logistic", "Raw logistic"),
                ("raw_mlp", "Raw MLP"),
            )
            if str(model).startswith(prefix)
        )
    )
    positive_budgets = sorted(
        int(value) for value in cells.label_budget_per_class.unique() if value > 0
    )
    validation = (
        cells[(cells.split == "val") & (cells.label_budget_per_class > 0)]
        .groupby(
            ["family", "target", "label_budget_per_class", "head_updates"],
            as_index=False,
        )
        .agg(validation_roc_auc=("roc_auc", "mean"))
    )
    choices = []
    for target in TARGET_ORDER:
        development = validation[validation.target != target]
        for budget in positive_budgets:
            winner = (
                development[development.label_budget_per_class == budget]
                .groupby("head_updates", as_index=False)
                .agg(development_validation_roc_auc=("validation_roc_auc", "mean"))
                .sort_values(
                    ["development_validation_roc_auc", "head_updates"],
                    ascending=[False, True],
                )
                .iloc[0]
            )
            choices.append(
                {
                    "target": target,
                    "label_budget_per_class": budget,
                    "selected_head_updates": int(winner.head_updates),
                }
            )
    choices = pd.DataFrame(choices)
    test = cells[(cells.split == "test") & (cells.label_budget_per_class > 0)]
    selected = test.merge(
        choices,
        left_on=["target", "label_budget_per_class", "head_updates"],
        right_on=["target", "label_budget_per_class", "selected_head_updates"],
        validate="many_to_one",
    )
    expected = cells.model_id.nunique() * len(TARGET_ORDER) * cells.label_seed.nunique() * len(
        positive_budgets
    )
    if len(selected) != expected:
        raise ValueError(f"extended selected grid has {len(selected)} rows, expected {expected}")
    return selected


def plot_by_target() -> None:
    """Facet the primary leakage-safe result so graph heterogeneity stays visible."""
    cells = extended_cross_target_selected()
    budgets = np.asarray(sorted(cells.label_budget_per_class.unique()))
    summary = (
        cells.groupby(["target", "family", "label_budget_per_class"], as_index=False)
        .agg(test_roc_auc_mean=("roc_auc", "mean"))
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8), sharex=True, sharey=True)
    x = np.arange(len(budgets))
    for panel, (axis, target) in enumerate(zip(axes.flat, TARGET_ORDER)):
        target_rows = summary[summary.target == target]
        for family in ORDER:
            rows = target_rows[target_rows.family == family].set_index(
                "label_budget_per_class"
            )
            values = [rows.loc[budget, "test_roc_auc_mean"] for budget in budgets]
            axis.plot(
                x,
                values,
                color=COLORS[family],
                marker=MARKERS[family],
                linewidth=2,
                markersize=6,
                label=family,
            )
        axis.set_title(f"{chr(ord('a') + panel)}  {TARGET_LABELS[target]}", loc="left")
        axis.set_xticks(x, [str(value) for value in budgets])
        axis.grid(alpha=0.22)
    for axis in axes[-1]:
        axis.set_xlabel("Labeled examples per class")
    for axis in axes[:, 0]:
        axis.set_ylabel("Test ROC-AUC")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.suptitle(
        "Frozen-encoder label efficiency by unseen target graph",
        y=1.07,
    )
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"label_efficiency_by_target.{extension}",
            dpi=220 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    summary.to_csv(DATA / "label_efficiency_by_target.csv", index=False)


def plot_combined_extended() -> None:
    """Aggregate the extended leakage-safe result across the four target graphs."""
    cells = extended_cross_target_selected()
    summary = (
        cells.groupby(["family", "label_budget_per_class"], as_index=False)
        .agg(test_roc_auc_mean=("roc_auc", "mean"))
    )
    budgets = np.asarray(sorted(summary.label_budget_per_class.unique()))
    x = np.arange(len(budgets))
    fig, axis = plt.subplots(figsize=(7.4, 5.0))
    for family in ORDER:
        rows = summary[summary.family == family].set_index("label_budget_per_class")
        axis.plot(
            x,
            [rows.loc[budget, "test_roc_auc_mean"] for budget in budgets],
            color=COLORS[family],
            marker=MARKERS[family],
            linewidth=2.2,
            markersize=7,
            label=family,
        )
    axis.set_xticks(x, [str(value) for value in budgets])
    axis.set_xlabel("Labeled examples per class")
    axis.set_ylabel("Mean test ROC-AUC across target graphs")
    axis.set_title("Frozen-encoder label efficiency across unseen target graphs")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"label_efficiency_combined_extended.{extension}",
            dpi=220 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    summary.to_csv(DATA / "label_efficiency_combined_extended.csv", index=False)


def main() -> int:
    primary = pd.read_csv(DATA / "cross_target_selected_summary.csv")
    fixed = pd.read_csv(DATA / "fixed_update_100_label_efficiency.csv")
    primary = primary[primary.label_budget_per_class > 0]
    fixed = fixed[fixed.label_budget_per_class > 0]
    joined = primary.merge(
        fixed,
        on=["family", "label_budget_per_class"],
        suffixes=("_primary", "_fixed100"),
        validate="one_to_one",
    )
    joined["primary_minus_fixed100"] = (
        joined.test_roc_auc_mean - joined.roc_auc_mean
    )

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12.2, 4.8), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    x = np.arange(len(BUDGETS))
    for family in ORDER:
        rows = primary[primary.family == family].set_index("label_budget_per_class")
        values = [rows.loc[budget, "test_roc_auc_mean"] for budget in BUDGETS]
        left.plot(
            x,
            values,
            color=COLORS[family],
            marker=MARKERS[family],
            linewidth=2,
            markersize=7,
            label=family,
        )
    left.set_xticks(x, [str(value) for value in BUDGETS])
    left.set_xlabel("Labeled examples per class")
    left.set_ylabel("Test ROC-AUC")
    left.set_title("a  Primary cross-target-selected performance", loc="left")
    left.grid(alpha=0.22)

    offsets = np.linspace(-0.30, 0.30, len(ORDER))
    for offset, family in zip(offsets, ORDER):
        rows = joined[joined.family == family].set_index("label_budget_per_class")
        values = [rows.loc[budget, "primary_minus_fixed100"] for budget in BUDGETS]
        right.plot(
            x + offset,
            values,
            linestyle="none",
            marker=MARKERS[family],
            color=COLORS[family],
            markersize=7,
        )
        for xpos, value in zip(x + offset, values):
            right.vlines(xpos, 0, value, color=COLORS[family], alpha=0.45, linewidth=1.3)
    right.axhline(0, color="0.25", linewidth=1)
    right.set_xticks(x, [str(value) for value in BUDGETS])
    right.set_xlabel("Labeled examples per class")
    right.set_ylabel("Primary minus fixed-100 ROC-AUC")
    right.set_title("b  Selection effect relative to fixed-100 endpoint", loc="left")
    right.grid(axis="y", alpha=0.22)

    handles, labels = left.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle("Frozen-encoder adaptation under a leakage-safe selection protocol", y=1.10)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"selection_protocol_comparison.{extension}",
            dpi=220 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    joined.to_csv(DATA / "selection_protocol_comparison.csv", index=False)
    plot_by_target()
    plot_combined_extended()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
