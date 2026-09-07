#!/usr/bin/env python3
"""Plot the five-label-seed RQ1 grid, including held-out Facebook."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUTS = (
    HERE / "data/social4_seed0_5label_raw",
    HERE / "data/facebook_seed0_5label_raw",
)
TARGETS = (
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
    "facebook_page_category_top30",
)
LABELS = {
    "covid_political": "COVID political",
    "election2020": "Election 2020",
    "ukr_rus_suspended": "Ukraine suspended",
    "twibot20": "TwiBot-20",
    "facebook_page_category_top30": "Facebook page category (30-class)",
}
COLORS = {"scratch": "#777777", "pretrained": "#2878B5"}
MARKERS = {"scratch": "s", "pretrained": "o"}


def load() -> pd.DataFrame:
    rows = []
    for root in INPUTS:
        for path in root.rglob("result.json"):
            value = json.loads(path.read_text())
            rows.append(
                {
                    "target": value["target"],
                    "budget": int(value["budget_per_class"]),
                    "arm": value["arm"],
                    "label_seed": int(value.get("label_seed", value["seed"])),
                    "test_auc": float(value["test"]["roc_auc"]),
                }
            )
    frame = pd.DataFrame(rows)
    expected = 5 * 4 * 2 * 5
    if len(frame) != expected:
        raise ValueError(f"expected {expected} cells, found {len(frame)}")
    if frame.duplicated(["target", "budget", "arm", "label_seed"]).any():
        raise ValueError("duplicate result cells")
    return frame


def main() -> int:
    frame = load()
    summary = frame.groupby(["target", "budget", "arm"], as_index=False).agg(
        mean=("test_auc", "mean"), std=("test_auc", "std")
    )
    out_data = HERE / "data/rq1_seed0_5label_with_facebook_summary.csv"
    summary.to_csv(out_data, index=False)

    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.2))
    axes = axes.flat
    for index, target in enumerate(TARGETS):
        axis = axes[index]
        target_rows = summary[summary.target == target]
        budgets = sorted(target_rows.budget.unique())
        x = np.arange(len(budgets))
        values = []
        for arm in ("scratch", "pretrained"):
            rows = target_rows[target_rows.arm == arm].set_index("budget").loc[budgets]
            mean = rows["mean"].to_numpy()
            std = rows["std"].to_numpy()
            values.extend((mean - std).tolist())
            values.extend((mean + std).tolist())
            axis.errorbar(
                x,
                mean,
                yerr=std,
                color=COLORS[arm],
                marker=MARKERS[arm],
                linewidth=2,
                capsize=3,
                label="Multi-graph SSL" if arm == "pretrained" else "Scratch",
            )
        low, high = min(values), max(values)
        margin = max(0.015, (high - low) * 0.16)
        axis.set_ylim(max(0.0, low - margin), min(1.0, high + margin))
        axis.set_xticks(x, [str(value) for value in budgets])
        axis.set_title(f"{chr(97 + index)}  {LABELS[target]}", loc="left")
        axis.set_xlabel("Labels per class")
        axis.set_ylabel("Test ROC-AUC")
        axis.grid(alpha=0.22)
    axes[5].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)
    fig.suptitle("Label-efficient adaptation to unseen target graphs (model seed 0, five label seeds)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = HERE / "figures/rq1_seed0_test_label_efficiency_per_target.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
