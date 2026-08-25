#!/usr/bin/env python3
"""Validate and plot the VISION native mixture-diversity CLS trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.experiments.setup.vision_native_mixture_finalcore.mixture_plan import (
    RUNGS,
    build_mixture_models,
)


ROOT = Path(__file__).resolve().parent
CHECKPOINTS = (100, 300, 900, 2500)
TARGETS = (
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
    "facebook_page_reference",
)
COLORS = {"A": "#4477AA", "B": "#CC6677", "C": "#228833"}


def read_jsonl_tree(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return pd.DataFrame(rows)


def load_cells(new_root: Path, all9_root: Path) -> pd.DataFrame:
    new = read_jsonl_tree(new_root)
    all9 = read_jsonl_tree(all9_root)
    all9 = all9[(all9["model_id"] == "all9") & (all9["training_seed"] == 0)]
    expected_new = {model.model_id for model in build_mixture_models()} - {"all9"}
    if set(new["model_id"]) != expected_new:
        raise ValueError(
            f"VISION mixture model registry mismatch: {sorted(set(new['model_id']))}"
        )
    frame = pd.concat((new, all9), ignore_index=True)
    if set(frame["architecture"]) != {"vision"} or set(frame["task"]) != {"classification"}:
        raise ValueError("mixture input contains non-VISION or non-CLS rows")
    if set(frame["training_seed"]) != {0}:
        raise ValueError("registered VISION mixture study requires training seed 0")
    if set(frame["checkpoint_step"]) != set(CHECKPOINTS):
        raise ValueError("VISION mixture checkpoints are incomplete")
    if set(frame["dataset"]) != set(TARGETS):
        raise ValueError("VISION mixture target panel changed")
    keys = ["model_id", "checkpoint_step", "dataset", "training_seed"]
    if frame.duplicated(keys).any() or len(frame) != 13 * 4 * 5:
        raise ValueError(f"expected 260 unique mixture cells, got {len(frame)}")
    fingerprints = frame.groupby("dataset")["episode_fingerprint"].nunique()
    if not (fingerprints == 1).all():
        raise ValueError(f"downstream episode drift: {fingerprints.to_dict()}")
    return frame


def expand_orders(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        model.model_id: model.aliases for model in build_mixture_models()
    }
    rows = []
    for row in frame.to_dict("records"):
        for alias in aliases[row["model_id"]]:
            _, order, rung_text = alias.split(":")
            rows.append({**row, "order": order, "rung": int(rung_text)})
    expanded = pd.DataFrame(rows)
    expected = 3 * len(RUNGS) * len(CHECKPOINTS) * len(TARGETS)
    if len(expanded) != expected:
        raise ValueError(f"expected {expected} order-expanded cells, got {len(expanded)}")
    return expanded


def plot_trajectory(summary: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
    for axis, step in zip(axes, CHECKPOINTS):
        subset = summary[summary.checkpoint_step == step]
        for order in ("A", "B", "C"):
            line = subset[subset.order == order].sort_values("rung")
            axis.plot(
                line.rung, line.roc_auc, marker="o", linewidth=2,
                color=COLORS[order], label=f"order {order}",
            )
        axis.set_title(f"{step:,} updates")
        axis.set_xticks(RUNGS)
        axis.set_xlabel("source graphs")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("mean downstream ROC-AUC\n(five fixed CLS targets)")
    axes[-1].legend(frameon=False, loc="best")
    figure.suptitle("VISION native feature-similarity mixture diversity (fixed compute)")
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_terminal_targets(expanded: pd.DataFrame, output: Path) -> None:
    terminal = expanded[expanded.checkpoint_step == 2500]
    figure, axes = plt.subplots(2, 3, figsize=(11, 7), sharex=True, sharey=True)
    display = {
        "covid_political": "COVID Political",
        "election2020": "Election 2020",
        "ukr_rus_suspended": "Ukraine Suspended",
        "twibot20": "TwiBot-20",
        "facebook_page_reference": "Facebook Page",
    }
    for axis, target in zip(axes.flat, TARGETS):
        subset = terminal[terminal.dataset == target]
        for order in ("A", "B", "C"):
            line = subset[subset.order == order].sort_values("rung")
            axis.plot(line.rung, line.roc_auc, marker="o", color=COLORS[order], label=order)
        axis.set_title(display[target])
        axis.set_xticks(RUNGS)
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    axes[0, 0].set_ylabel("ROC-AUC")
    axes[1, 0].set_ylabel("ROC-AUC")
    for axis in axes[1, :2]:
        axis.set_xlabel("source graphs")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, [f"order {value}" for value in labels], frameon=False)
    figure.suptitle("VISION mixture diversity at the 2,500-update endpoint")
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-root", type=Path,
        default=ROOT / "data" / "vision_native_mixture_raw",
    )
    parser.add_argument(
        "--all9-root", type=Path,
        default=ROOT / "data" / "vision_all9_saturation_raw",
    )
    args = parser.parse_args()
    frame = load_cells(args.new_root, args.all9_root)
    expanded = expand_orders(frame)
    per_target = (
        expanded.groupby(["checkpoint_step", "order", "rung", "dataset"], as_index=False)
        .agg(roc_auc=("roc_auc", "mean"), accuracy=("accuracy", "mean"), f1=("f1", "mean"))
    )
    aggregate = (
        per_target.groupby(["checkpoint_step", "order", "rung"], as_index=False)
        .agg(roc_auc=("roc_auc", "mean"), accuracy=("accuracy", "mean"), f1=("f1", "mean"))
    )
    expanded.to_csv(ROOT / "data" / "vision_native_mixture_cells.csv", index=False)
    per_target.to_csv(ROOT / "data" / "vision_native_mixture_per_target.csv", index=False)
    aggregate.to_csv(ROOT / "data" / "vision_native_mixture_summary.csv", index=False)
    plot_trajectory(aggregate, ROOT / "figures" / "vision_mixture_diversity_trajectory")
    plot_terminal_targets(expanded, ROOT / "figures" / "vision_mixture_diversity_terminal_targets")
    print("VISION_MIXTURE_OK physical_cells=260 order_expanded_cells=300")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
