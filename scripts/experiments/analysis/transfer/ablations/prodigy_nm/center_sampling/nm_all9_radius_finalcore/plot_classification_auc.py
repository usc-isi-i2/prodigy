#!/usr/bin/env python3
"""Aggregate radius CLS shards and plot ROC-AUC checkpoint trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


HERE = Path(__file__).resolve().parent
ARMS = ("global", "radius_mix", "close_only")
STEPS = (100, 300, 900, 2500)
TARGETS = (
    "covid_political",
    "election2020",
    "facebook_page_reference",
    "twibot20",
    "ukr_rus_suspended",
)
LABELS = {
    "global": "Global",
    "radius_mix": "Radius mix",
    "close_only": "Close only",
    "covid_political": "COVID political",
    "election2020": "Election 2020",
    "facebook_page_reference": "Facebook pages",
    "twibot20": "TwiBot-20",
    "ukr_rus_suspended": "UKR–RUS suspended",
    "macro_mean": "Five-target macro mean",
}
COLORS = {"global": "#0072B2", "radius_mix": "#D55E00", "close_only": "#009E73"}
MARKERS = {"global": "o", "radius_mix": "s", "close_only": "^"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--data-output", type=Path, default=HERE / "data" / "classification_trajectory.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures")
    return parser.parse_args()


def load_rows(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    data = pd.DataFrame(rows)
    required = {"model_id", "training_seed", "checkpoint_step", "dataset", "roc_auc"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    keys = set(data[["model_id", "training_seed", "checkpoint_step", "dataset"]].itertuples(index=False, name=None))
    expected = {(arm, seed, step, target) for arm in ARMS for seed in range(3) for step in STEPS for target in TARGETS}
    if keys != expected or len(data) != len(expected):
        raise ValueError(f"coverage mismatch: rows={len(data)}/{len(expected)}, missing={len(expected - keys)}")
    return data.rename(columns={"model_id": "arm", "roc_auc": "auc"})


def plot_panel(ax, data: pd.DataFrame, target: str) -> None:
    subset = data if target == "macro_mean" else data[data.dataset == target]
    for arm in ARMS:
        arm_data = subset[subset.arm == arm]
        if target == "macro_mean":
            arm_data = arm_data.groupby(["training_seed", "checkpoint_step"], as_index=False).auc.mean()
        pivot = arm_data.pivot(index="checkpoint_step", columns="training_seed", values="auc").reindex(STEPS)
        for seed in pivot.columns:
            ax.plot(STEPS, pivot[seed], color=COLORS[arm], alpha=0.18, linewidth=0.9)
        mean = pivot.mean(axis=1)
        ax.fill_between(STEPS, pivot.min(axis=1), pivot.max(axis=1), color=COLORS[arm], alpha=0.08, linewidth=0)
        ax.plot(STEPS, mean, color=COLORS[arm], marker=MARKERS[arm], linewidth=2.1, markersize=4.8,
                markeredgecolor="white", markeredgewidth=0.6)
    ax.axhline(0.5, color="#666666", linestyle=(0, (3, 3)), linewidth=0.8, alpha=0.7)
    ax.set_xscale("log")
    ax.set_xticks(STEPS, [str(step) for step in STEPS])
    ax.set_title(LABELS[target])
    ax.set_ylim(0.45, 1.01)
    ax.grid(axis="y", color="#B8B8B8", linewidth=0.6, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> int:
    args = parse_args()
    data = load_rows(args.results_root)
    columns = ["arm", "training_seed", "checkpoint_step", "dataset", "auc", "accuracy", "f1", "episode_fingerprint"]
    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    data[columns].sort_values(columns[:4]).to_csv(args.data_output, index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 11, "axes.labelsize": 10})
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True, sharey=True)
    for ax, target in zip(axes.flat, (*TARGETS, "macro_mean")):
        plot_panel(ax, data, target)
    for ax in axes[1]:
        ax.set_xlabel("Completed optimizer updates")
    for ax in axes[:, 0]:
        ax.set_ylabel("Classification ROC-AUC")
    handles = [Line2D([0], [0], color=COLORS[a], marker=MARKERS[a], linewidth=2.1, label=LABELS[a]) for a in ARMS]
    handles.append(Line2D([0], [0], color="#666666", linestyle=(0, (3, 3)), linewidth=0.8, label="Chance"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("Radius-controlled NM pretraining: downstream classification trajectories", fontsize=14, y=0.99)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.09, top=0.85, hspace=0.25, wspace=0.08)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"classification_auc_trajectories.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(args.data_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
