#!/usr/bin/env python3
"""Plot available validation trajectories without inventing missing cells."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "data" / "validation_trajectory.csv"
DEFAULT_OUTPUT_DIR = HERE / "figures"

ARM_ORDER = ("global", "radius_mix", "close_only")
PANEL_ORDER = ("radius2", "radius3", "global", "within_source")

ARM_LABELS = {
    "global": "Global",
    "radius_mix": "Radius mix",
    "close_only": "Close only",
}
PANEL_LABELS = {
    "radius2": "Radius 2 evaluation",
    "radius3": "Radius 3 evaluation",
    "global": "Global evaluation",
    "within_source": "Within-source evaluation",
}
COLORS = {
    "global": "#0072B2",
    "radius_mix": "#D55E00",
    "close_only": "#009E73",
}
MARKERS = {
    "global": "o",
    "radius_mix": "s",
    "close_only": "^",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default="validation_trajectories")
    parser.add_argument(
        "--title",
        default="All-nine radius experiment: validation trajectories",
    )
    return parser.parse_args()


def validate(data: pd.DataFrame) -> None:
    required = {
        "arm",
        "seed",
        "checkpoint_step",
        "panel",
        "score",
        "loss",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    keys = list(
        data[["arm", "seed", "checkpoint_step", "panel"]]
        .itertuples(index=False, name=None)
    )
    if len(keys) != len(set(keys)):
        raise ValueError("input contains duplicate arm x seed x checkpoint x panel cells")
    if data.empty:
        raise ValueError("input contains no trajectories")


def plot_metric(
    ax,
    data: pd.DataFrame,
    panel: str,
    metric: str,
    arms: tuple[str, ...],
    steps: tuple[int, ...],
) -> None:
    panel_data = data[data["panel"] == panel]
    for arm in arms:
        arm_data = panel_data[panel_data["arm"] == arm]
        if arm_data.empty:
            continue
        pivot = arm_data.pivot(
            index="checkpoint_step", columns="seed", values=metric
        ).reindex(steps)

        for seed in pivot.columns:
            series = pivot[seed].dropna()
            ax.plot(
                series.index,
                series,
                color=COLORS[arm],
                linewidth=0.9,
                alpha=0.24,
                marker=MARKERS[arm],
                markersize=3.2,
                markeredgewidth=0,
                zorder=1,
            )

        mean = pivot.mean(axis=1).dropna()
        if len(pivot.columns) > 1:
            low = pivot.min(axis=1).reindex(mean.index)
            high = pivot.max(axis=1).reindex(mean.index)
            ax.fill_between(
                mean.index,
                low.to_numpy(),
                high.to_numpy(),
                color=COLORS[arm],
                alpha=0.08,
                linewidth=0,
                zorder=0,
            )
        ax.plot(
            mean.index,
            mean,
            color=COLORS[arm],
            linewidth=2.3,
            marker=MARKERS[arm],
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=3,
        )

    if max(steps) / min(steps) > 5:
        ax.set_xscale("log")
    ax.set_xticks(steps, [f"{step:,}" for step in steps])
    ax.grid(axis="y", color="#B8B8B8", linewidth=0.6, alpha=0.35)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)


def main() -> int:
    args = parse_args()
    data = pd.read_csv(args.input)
    validate(data)
    arms = tuple(arm for arm in ARM_ORDER if arm in set(data["arm"]))
    panels = tuple(panel for panel in PANEL_ORDER if panel in set(data["panel"]))
    steps = tuple(sorted(int(step) for step in data["checkpoint_step"].unique()))
    unknown_arms = sorted(set(data["arm"]) - set(arms))
    unknown_panels = sorted(set(data["panel"]) - set(panels))
    if unknown_arms or unknown_panels:
        raise ValueError(
            f"unknown arms or panels: arms={unknown_arms}, panels={unknown_panels}"
        )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "normal",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "figure.dpi": 140,
        }
    )
    fig, axes = plt.subplots(
        2,
        len(panels),
        figsize=(4.1 * len(panels), 7.4),
        sharex="col",
        sharey="row",
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        bottom=0.09,
        top=0.84,
        hspace=0.06,
        wspace=0.04,
    )

    for column, panel in enumerate(panels):
        plot_metric(axes[0, column], data, panel, "score", arms, steps)
        plot_metric(axes[1, column], data, panel, "loss", arms, steps)
        axes[0, column].set_title(PANEL_LABELS[panel], pad=9)
        axes[1, column].set_xlabel("Completed optimizer updates")

    axes[0, 0].set_ylabel("Validation accuracy")
    axes[1, 0].set_ylabel("Validation cross-entropy loss")
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    for ax in axes[0]:
        ax.axhline(
            1 / 30,
            color="#666666",
            linewidth=0.8,
            linestyle=(0, (3, 3)),
            alpha=0.65,
            zorder=-1,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[arm],
            marker=MARKERS[arm],
            markeredgecolor="white",
            markeredgewidth=0.7,
            linewidth=2.3,
            markersize=6,
            label=ARM_LABELS[arm],
        )
        for arm in arms
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#666666",
            linestyle=(0, (3, 3)),
            linewidth=0.8,
            label="30-way chance",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=len(arms) + 1,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        args.title,
        fontsize=14,
        fontweight="normal",
        y=0.985,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / f"{args.basename}.png"
    pdf_path = args.output_dir / f"{args.basename}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(png_path)
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
