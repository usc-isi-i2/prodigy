#!/usr/bin/env python3
"""Merge early 2.5k and late 10k validation trajectories without joining runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


HERE = Path(__file__).resolve().parent
DEFAULT_EARLY = HERE / "data" / "validation_trajectory.csv"
DEFAULT_LATE = HERE / "data" / "validation_trajectory_10k_available.csv"
DEFAULT_OUTPUT_DIR = HERE / "figures"

ARMS = ("global", "radius_mix")
PANELS = ("radius2", "radius3", "global", "within_source")
ARM_LABELS = {"global": "Global", "radius_mix": "Radius mix"}
PANEL_LABELS = {
    "radius2": "Radius 2 evaluation",
    "radius3": "Radius 3 evaluation",
    "global": "Global evaluation",
    "within_source": "Within-source evaluation",
}
COLORS = {"global": "#0072B2", "radius_mix": "#D55E00"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early-input", type=Path, default=DEFAULT_EARLY)
    parser.add_argument("--late-input", type=Path, default=DEFAULT_LATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_segments(early_path: Path, late_path: Path) -> pd.DataFrame:
    early = pd.read_csv(early_path)
    early = early[
        early["arm"].isin(ARMS)
        & (early["seed"] == 0)
        & (early["checkpoint_step"] < 2500)
    ].copy()
    early["run_segment"] = "original_2p5k"

    late = pd.read_csv(late_path)
    late = late[late["arm"].isin(ARMS) & (late["seed"] == 0)].copy()
    late["run_segment"] = "new_10k"

    data = pd.concat([early, late], ignore_index=True)
    keys = ["arm", "seed", "checkpoint_step", "panel", "run_segment"]
    if data.duplicated(keys).any():
        raise ValueError("duplicate trajectory cells")
    return data


def plot_metric(ax, data: pd.DataFrame, panel: str, metric: str) -> None:
    panel_data = data[data["panel"] == panel]
    for arm in ARMS:
        arm_data = panel_data[panel_data["arm"] == arm]
        for segment, linestyle, marker, fillstyle in (
            ("original_2p5k", "--", "o", "none"),
            ("new_10k", "-", "o", "full"),
        ):
            series = (
                arm_data[arm_data["run_segment"] == segment]
                .sort_values("checkpoint_step")
            )
            if series.empty:
                continue
            ax.plot(
                series["checkpoint_step"],
                series[metric],
                color=COLORS[arm],
                linestyle=linestyle,
                linewidth=2.1,
                marker=marker,
                markersize=4.5,
                markerfacecolor=("none" if fillstyle == "none" else COLORS[arm]),
                markeredgecolor=COLORS[arm],
                markeredgewidth=1.1,
                zorder=3,
            )

    ax.set_xscale("log")
    steps = (100, 300, 900, 2500, 5000, 7500, 10000)
    ax.set_xticks(steps, ["100", "300", "900", "2.5k", "5k", "7.5k", "10k"])
    ax.axvspan(900, 2500, color="#7A7A7A", alpha=0.045, linewidth=0)
    ax.grid(axis="y", color="#B8B8B8", linewidth=0.6, alpha=0.35)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> int:
    args = parse_args()
    data = load_segments(args.early_input, args.late_input)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16.4, 7.4),
        sharex="col",
        sharey="row",
    )
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.11, top=0.82, hspace=0.12, wspace=0.05)

    for column, panel in enumerate(PANELS):
        plot_metric(axes[0, column], data, panel, "score")
        plot_metric(axes[1, column], data, panel, "loss")
        axes[0, column].set_title(PANEL_LABELS[panel], pad=9)
        axes[1, column].set_xlabel("Completed optimizer updates")

    axes[0, 0].set_ylabel("Validation accuracy")
    axes[1, 0].set_ylabel("Validation cross-entropy loss")
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    for ax in axes[0]:
        ax.axhline(1 / 30, color="#666666", linestyle=":", linewidth=1.0, alpha=0.75)

    handles = [
        Line2D([0], [0], color=COLORS[arm], linewidth=2.3, marker="o", markersize=5, label=ARM_LABELS[arm])
        for arm in ARMS
    ]
    handles.extend(
        [
            Line2D([0], [0], color="#666666", linestyle="--", marker="o", markerfacecolor="none", linewidth=1.8, label="Original 2.5k run (100–900)"),
            Line2D([0], [0], color="#666666", linestyle="-", marker="o", linewidth=1.8, label="New 10k rerun (2.5k–10k)"),
            Line2D([0], [0], color="#666666", linestyle=":", linewidth=1.0, label="30-way chance"),
        ]
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.915), ncol=5, frameon=False, fontsize=9.5)
    fig.suptitle("All-nine radius experiment: merged validation trajectories", fontsize=14, fontweight="normal", y=0.985)
    fig.text(0.5, 0.945, "The gap marks a restart: early and late points come from separate seed-0 training runs.", ha="center", va="top", fontsize=10, color="#555555")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "merged_validation_trajectories_available.png"
    pdf_path = args.output_dir / "merged_validation_trajectories_available.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(png_path)
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
