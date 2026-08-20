#!/usr/bin/env python3
"""Plot step-2,500 CLS and NM ROC-AUC panels for the radius arms."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ARMS = ("global", "radius_mix", "close_only")
ARM_LABELS = ("Global", "Radius mix", "Close only")
COLORS = ("#0072B2", "#D55E00", "#009E73")
CLS_TARGETS = (
    "covid_political", "election2020", "facebook_page_reference",
    "twibot20", "ukr_rus_suspended",
)
NM_PANELS = ("radius2", "radius3", "global", "within_source")
LABELS = {
    "covid_political": "COVID political",
    "election2020": "Election 2020",
    "facebook_page_reference": "Facebook pages",
    "twibot20": "TwiBot-20",
    "ukr_rus_suspended": "UKR–RUS suspended",
    "radius2": "Radius 2",
    "radius3": "Radius 3",
    "global": "Global episodes",
    "within_source": "Within source",
    "macro": "Macro mean",
    "all_macro": "All-panel macro mean",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification", type=Path, default=HERE / "data" / "classification_trajectory.csv")
    parser.add_argument("--nm", type=Path, required=True)
    parser.add_argument("--nm-output", type=Path, default=HERE / "data" / "nm_step2500_auc.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE / "figures")
    return parser.parse_args()


def add_macro(data: pd.DataFrame, group_col: str, members: tuple[str, ...], label: str) -> pd.DataFrame:
    macro = (
        data[data[group_col].isin(members)]
        .groupby(["arm", "training_seed"], as_index=False).auc.mean()
        .assign(**{group_col: label})
    )
    return pd.concat([data, macro], ignore_index=True)


def plot_panel(ax, data: pd.DataFrame, group_col: str, panel: str, ylabel: str | None = None) -> None:
    subset = data[data[group_col] == panel]
    x = np.arange(len(ARMS))
    offsets = (-0.07, 0.0, 0.07)
    for index, (arm, color) in enumerate(zip(ARMS, COLORS)):
        values = subset[subset.arm == arm].sort_values("training_seed").auc.to_numpy()
        if len(values) != 3:
            raise ValueError(f"expected three seeds for {group_col}={panel}, arm={arm}")
        for seed, value in enumerate(values):
            ax.scatter(x[index] + offsets[seed], value, s=22, color=color, alpha=0.38, zorder=2)
        mean = float(values.mean())
        ax.scatter(x[index], mean, s=78, color=color, edgecolor="white", linewidth=0.8, zorder=4)
        ax.vlines(x[index], values.min(), values.max(), color=color, alpha=0.28, linewidth=5, zorder=1)
    ax.axhline(0.5, color="#777777", linestyle=(0, (3, 3)), linewidth=0.8, alpha=0.7)
    ax.set_title(LABELS[panel], fontsize=10.5)
    ax.set_xticks(x, ARM_LABELS, rotation=18, ha="right")
    ax.set_ylim(0.45, 1.01)
    ax.grid(axis="y", color="#B8B8B8", linewidth=0.6, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)


def main() -> int:
    args = parse_args()
    cls = pd.read_csv(args.classification)
    cls = cls[cls.checkpoint_step == 2500].copy()
    cls = add_macro(cls, "dataset", CLS_TARGETS, "macro")

    nm = pd.read_csv(args.nm)
    expected = {(arm, seed, panel) for arm in ARMS for seed in range(3) for panel in NM_PANELS}
    observed = set(nm[["arm", "training_seed", "panel"]].itertuples(index=False, name=None))
    if observed != expected or len(nm) != len(expected):
        raise ValueError("NM input must contain all 36 arm × seed × panel cells")
    nm = add_macro(nm, "panel", NM_PANELS[:3], "macro")
    nm = add_macro(nm, "panel", NM_PANELS, "all_macro")
    args.nm_output.parent.mkdir(parents=True, exist_ok=True)
    nm.sort_values(["panel", "arm", "training_seed"]).to_csv(args.nm_output, index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.labelsize": 10, "xtick.labelsize": 8.5})
    fig, axes = plt.subplots(4, 3, figsize=(12.8, 13.6), sharey=True)
    cls_panels = (*CLS_TARGETS, "macro")
    nm_panels = (*NM_PANELS, "macro", "all_macro")
    for index, (ax, panel) in enumerate(zip(axes[:2].flat, cls_panels)):
        plot_panel(ax, cls, "dataset", panel, "CLS ROC-AUC" if index % 3 == 0 else None)
    for index, (ax, panel) in enumerate(zip(axes[2:].flat, nm_panels)):
        plot_panel(ax, nm, "panel", panel, "NM ROC-AUC" if index % 3 == 0 else None)
    fig.text(0.5, 0.955, "Downstream classification", ha="center", fontsize=13, weight="bold")
    fig.text(0.5, 0.49, "Neighbor matching", ha="center", fontsize=13, weight="bold")
    fig.suptitle("Radius-controlled NM pretraining at 2,500 optimizer updates", fontsize=15, y=0.995)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.06, top=0.93, hspace=0.52, wspace=0.10)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"step2500_cls_nm_auc.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
