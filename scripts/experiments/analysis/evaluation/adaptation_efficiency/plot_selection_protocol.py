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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
