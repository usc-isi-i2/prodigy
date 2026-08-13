#!/usr/bin/env python3
"""Plot the sequential and interleaved ladders plus their paired AUC deltas."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
DATASETS = [
    "ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
    "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter",
]
SHORT = ["ukr", "covid", "midterm", "cov_pol", "elec20", "ukr_susp", "twibot", "cp_hk"]
RUNG_LABELS = [
    "L1 ukr", "L2 +cov", "L3 +mid", "L4 +cov_pol",
    "L5 +elec20", "L6 +ukr_susp", "L7 +twibot", "L8 all8",
]
BLUES = LinearSegmentedColormap.from_list(
    "nm_blue", ["#E6F1FB", "#85B7EB", "#185FA5", "#0C447C"]
)
CORAL = "#D85A30"
BLUE = "#2a78d6"
INK = "#111111"
MUTED = "#898781"
GRID = "#e1e0d9"
RUNG_TICKS = [
    "ukr", "+covid", "+midterm", "+cov-pol", "+elec '20",
    "+ukr-susp", "+twibot", "+cp-hk\n(all 8)",
]
GRAPH_LABELS = [
    "Ukr-Rus", "COVID-19", "Midterm", "COVID-pol.",
    "Election '20", "Ukr-Rus susp.", "TwiBot-20", "CP-HK",
]
GRAY = "#8f8d87"


def load_matrices(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    interleaved = np.full((8, 8), np.nan)
    sequential = np.full((8, 8), np.nan)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            index = (int(row["rung"]) - 1, DATASETS.index(row["test_graph"]))
            interleaved[index] = float(row["auc_interleaved"])
            sequential[index] = float(row["auc_sequential"])
    if np.isnan(interleaved).any() or np.isnan(sequential).any():
        raise ValueError("expected a complete 8x8 schedule comparison")
    return interleaved, sequential, sequential - interleaved


def draw_ladder(ax, matrix: np.ndarray, title: str, *, vmin: float, vmax: float) -> None:
    for row in range(8):
        for column in range(8):
            value = matrix[row, column]
            scaled = float(np.clip((value - vmin) / (vmax - vmin), 0, 1))
            y = 7 - row
            ax.add_patch(
                Rectangle(
                    (column, y), 1, 1, facecolor=BLUES(scaled),
                    edgecolor="white", linewidth=1.2,
                )
            )
            ax.text(
                column + 0.5, y + 0.5, f"{value:.3f}"[1:],
                ha="center", va="center", fontsize=7.5,
                color="white" if scaled > 0.55 else "#20303f",
            )
        ax.add_patch(
            Rectangle(
                (row, 7 - row), 1, 1, fill=False,
                edgecolor=CORAL, linewidth=2.2, zorder=5,
            )
        )
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xticks(np.arange(8) + 0.5, SHORT, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(8) + 0.5, RUNG_LABELS[::-1], fontsize=8.6)
    ax.tick_params(length=0)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_ladders(
    interleaved: np.ndarray,
    sequential: np.ndarray,
    output: Path,
) -> None:
    vmin = min(0.55, float(np.nanmin([interleaved, sequential])))
    vmax = 0.985
    fig = plt.figure(figsize=(14.5, 6.1), dpi=180)
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, 0.035), wspace=0.24)
    axes = (fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]))
    colorbar_ax = fig.add_subplot(grid[0, 2])
    draw_ladder(axes[0], interleaved, "Balanced interleaved", vmin=vmin, vmax=vmax)
    draw_ladder(axes[1], sequential, "Blocked sequential", vmin=vmin, vmax=vmax)
    fig.suptitle(
        "NM graph ladder — presentation order changes retention",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.text(
        0.5, 0.965,
        "test ROC-AUC · 3-shot / 30-way · matched 40k steps · orange box = newest graph",
        ha="center", va="top", fontsize=9, color="#686762",
    )
    scalar = plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=BLUES)
    colorbar = fig.colorbar(scalar, cax=colorbar_ax)
    colorbar.set_label("ROC-AUC")
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.18, top=0.87)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output, bbox_inches="tight")
    print(f"wrote {output}")
    print(f"wrote {pdf_output}")
    plt.close(fig)


def declutter_labels(items, gap: float, lower: float, upper: float):
    items = sorted(items, key=lambda item: item["value"])
    for item in items:
        item["label_y"] = item["value"]
    for index in range(1, len(items)):
        if items[index]["label_y"] - items[index - 1]["label_y"] < gap:
            items[index]["label_y"] = items[index - 1]["label_y"] + gap
    if items[-1]["label_y"] > upper:
        items[-1]["label_y"] = upper
        for index in range(len(items) - 2, -1, -1):
            if items[index + 1]["label_y"] - items[index]["label_y"] < gap:
                items[index]["label_y"] = items[index + 1]["label_y"] - gap
    for item in items:
        item["label_y"] = min(max(item["label_y"], lower), upper)
    return items


def plot_trajectory(matrix: np.ndarray, output: Path) -> None:
    x = np.arange(8)
    lower = min(0.60, float(matrix.min()) - 0.02)
    fig, ax = plt.subplots(figsize=(10.6, 5.9), dpi=200)
    label_items = []
    for graph_index, label in enumerate(GRAPH_LABELS):
        values = matrix[:, graph_index]
        entry_index = graph_index
        for rung in range(7):
            in_training = rung + 1 >= entry_index
            ax.plot(
                x[rung : rung + 2], values[rung : rung + 2],
                color=BLUE if in_training else GRAY,
                linewidth=1.9 if in_training else 1.5,
                linestyle="-" if in_training else (0, (4, 2)),
                zorder=3, solid_capstyle="round",
            )
        for rung in range(8):
            in_training = rung >= entry_index
            if rung == entry_index and entry_index > 0:
                ax.scatter(
                    x[rung], values[rung], s=95, facecolor=BLUE,
                    edgecolor="white", linewidth=1.6, zorder=6,
                )
            else:
                ax.scatter(
                    x[rung], values[rung], s=24,
                    facecolor=BLUE if in_training else "white",
                    edgecolor=BLUE if in_training else GRAY,
                    linewidth=1.4, zorder=5,
                )
        entry_delta = (
            values[entry_index] - values[entry_index - 1]
            if entry_index > 0 else 0.0
        )
        text = f"{label}   +{entry_delta:.2f}" if entry_delta >= 0.03 else label
        label_items.append({"value": values[-1], "text": text})

    mean = matrix.mean(axis=1)
    ax.plot(
        x, mean, color=INK, linewidth=2.8, zorder=8, marker="s", markersize=6,
        markerfacecolor=INK, markeredgecolor="white", markeredgewidth=1.2,
    )
    ax.annotate(
        "mean (all 8)", (x[0] - 0.06, mean[0]), ha="right", va="center",
        fontsize=9, color=INK, fontweight="bold",
    )

    for item in declutter_labels(label_items, 0.023, lower + 0.012, 0.99):
        ax.plot(
            [x[-1] + 0.06, x[-1] + 0.28],
            [item["value"], item["label_y"]],
            color=MUTED, linewidth=0.6, zorder=2,
        )
        ax.annotate(
            item["text"], (x[-1] + 0.34, item["label_y"]),
            ha="left", va="center", fontsize=9.3, color=BLUE,
            fontweight="bold",
        )

    ax.set_xlim(-0.62, 8.75)
    ax.set_ylim(lower, 1.0)
    ax.set_xticks(x, RUNG_TICKS, fontsize=9.2)
    ax.set_xlabel(
        "SSL pre-training graph  (one source added per rung, merge grows to the right)",
        fontsize=10.5, color=INK,
    )
    ax.set_ylabel("NM ROC-AUC  (3-shot, 30-way)", fontsize=10.5, color=INK)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(
        "Blocked sequential ladder: the newest graph rises while earlier graphs decay",
        fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=26,
    )
    ax.text(
        0.0, 1.02,
        "NM  3-shot / 30-way  ·  matched step 40k  ·  blocked sequential sampling  ·  "
        "+d = gain at entry rung",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=MUTED,
    )
    legend = [
        Line2D(
            [0], [0], color=BLUE, linewidth=1.9, marker="o",
            markerfacecolor=BLUE, markeredgecolor="white", markersize=7,
            label="in training prefix",
        ),
        Line2D(
            [0], [0], color=GRAY, linewidth=1.5, linestyle=(0, (4, 2)),
            marker="o", markerfacecolor="white", markeredgecolor=GRAY,
            markersize=7, label="held out",
        ),
        Line2D(
            [0], [0], color=INK, linewidth=2.8, marker="s",
            markerfacecolor=INK, markeredgecolor="white", markersize=7,
            label="mean (all 8 graphs)",
        ),
    ]
    ax.legend(
        handles=legend, loc="lower right", frameon=False,
        fontsize=9, handlelength=2.4,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output, bbox_inches="tight")
    print(f"wrote {output}")
    print(f"wrote {pdf_output}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=HERE / "data" / "nm_ladder_schedule_comparison_long.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=HERE / "figures" / "sequential_minus_interleaved.png",
    )
    parser.add_argument(
        "--ladder-output", type=Path,
        default=HERE / "figures" / "sequential_vs_interleaved_ladder.png",
    )
    parser.add_argument(
        "--trajectory-output", type=Path,
        default=HERE / "figures" / "nm_ladder_sequential_trajectory.png",
    )
    args = parser.parse_args()

    interleaved, sequential, matrix = load_matrices(args.input)

    limit = max(0.01, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(8), [name.replace("_twitter", "") for name in DATASETS], rotation=40, ha="right")
    ax.set_yticks(range(8), range(1, 9))
    ax.set_xlabel("evaluation graph")
    ax.set_ylabel("ladder rung")
    ax.set_title("Blocked sequential − balanced interleaved (AUC)")
    fig.colorbar(image, ax=ax, label="Δ ROC-AUC")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"wrote {args.output}")
    plt.close(fig)
    plot_ladders(interleaved, sequential, args.ladder_output)
    plot_trajectory(sequential, args.trajectory_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
