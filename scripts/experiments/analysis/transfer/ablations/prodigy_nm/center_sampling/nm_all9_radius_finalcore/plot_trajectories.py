#!/usr/bin/env python3
"""Plot all validation accuracy and loss trajectories for the radius experiment."""

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

ARMS = ("global", "radius_mix", "close_only")
PANELS = ("radius2", "radius3", "global")
STEPS = (100, 300, 900, 2500)

ARM_LABELS = {
    "global": "Global",
    "radius_mix": "Radius mix",
    "close_only": "Close only",
}
PANEL_LABELS = {
    "radius2": "Radius 2 evaluation",
    "radius3": "Radius 3 evaluation",
    "global": "Global evaluation",
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
    expected = {
        (arm, seed, step, panel)
        for arm in ARMS
        for seed in (0, 1, 2)
        for step in STEPS
        for panel in PANELS
    }
    observed = set(
        data[["arm", "seed", "checkpoint_step", "panel"]]
        .itertuples(index=False, name=None)
    )
    if observed != expected or len(data) != len(expected):
        raise ValueError(
            "input must contain every arm x seed x checkpoint x panel cell exactly once"
        )


def plot_metric(ax, data: pd.DataFrame, panel: str, metric: str) -> None:
    panel_data = data[data["panel"] == panel]
    for arm in ARMS:
        arm_data = panel_data[panel_data["arm"] == arm]
        pivot = arm_data.pivot(
            index="checkpoint_step", columns="seed", values=metric
        ).reindex(STEPS)

        for seed in pivot.columns:
            ax.plot(
                STEPS,
                pivot[seed],
                color=COLORS[arm],
                linewidth=0.9,
                alpha=0.24,
                marker=MARKERS[arm],
                markersize=3.2,
                markeredgewidth=0,
                zorder=1,
            )

        mean = pivot.mean(axis=1)
        low = pivot.min(axis=1)
        high = pivot.max(axis=1)
        ax.fill_between(
            STEPS,
            low.to_numpy(),
            high.to_numpy(),
            color=COLORS[arm],
            alpha=0.08,
            linewidth=0,
            zorder=0,
        )
        ax.plot(
            STEPS,
            mean,
            color=COLORS[arm],
            linewidth=2.3,
            marker=MARKERS[arm],
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xticks(STEPS, [str(step) for step in STEPS])
    ax.grid(axis="y", color="#B8B8B8", linewidth=0.6, alpha=0.35)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)


def main() -> int:
    args = parse_args()
    data = pd.read_csv(args.input)
    validate(data)

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
        3,
        figsize=(13.5, 7.4),
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

    for column, panel in enumerate(PANELS):
        plot_metric(axes[0, column], data, panel, "score")
        plot_metric(axes[1, column], data, panel, "loss")
        axes[0, column].set_title(PANEL_LABELS[panel], pad=9)
        axes[1, column].set_xlabel("Completed optimizer updates")

    axes[0, 0].set_ylabel("Validation accuracy")
    axes[1, 0].set_ylabel("Validation cross-entropy loss")
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axes[0, 0].set_ylim(0.03, 0.66)
    axes[1, 0].set_ylim(1.05, 3.35)

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
        for arm in ARMS
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
        ncol=4,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "All-nine radius experiment: validation trajectories",
        fontsize=14,
        fontweight="normal",
        y=0.985,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "validation_trajectories.png"
    pdf_path = args.output_dir / "validation_trajectories.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(png_path)
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
