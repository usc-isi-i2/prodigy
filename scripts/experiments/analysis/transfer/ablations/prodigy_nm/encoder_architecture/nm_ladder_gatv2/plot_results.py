#!/usr/bin/env python3
"""Render the GATv2 NM-ladder result figures.

Outputs under ``figures/``:

* ``nm_ladder_gatv2_trajectory``: the complete eight-graph GATv2 staircase.
* ``nm_ladder_gatv2_backbone_comparison``: entry-jump agreement and all-cell
  parity against the matched GraphSAGE ladder.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"
GATV2_CSV = DATA / "nm_ladder_gatv2.csv"
COMPARISON_CSV = DATA / "nm_ladder_backbone_comparison.csv"

GRAPHS = [
    ("ukr_rus_twitter", "Ukr-Rus", 1),
    ("covid19_twitter", "COVID-19", 2),
    ("midterm", "Midterm", 3),
    ("covid_political", "COVID-pol.", 4),
    ("election2020", "Election '20", 5),
    ("ukr_rus_suspended", "Ukr-Rus susp.", 6),
    ("twibot20", "TwiBot-20", 7),
    ("cp_hk_twitter", "CP-HK", 8),
]
CANON = [key for key, _, _ in GRAPHS]
RUNGS = np.arange(1, 9)
XTICKS = [
    "ukr",
    "+covid",
    "+midterm",
    "+cov-pol",
    "+elec '20",
    "+ukr-susp",
    "+twibot",
    "+cp-hk\n(all 8)",
]

BLUE = "#2A78D6"
BLUE_DARK = "#174A7E"
CORAL = "#D85A30"
GRAY = "#8F8D87"
INK = "#0B0B0B"
MUTED = "#6F6D68"
GRID = "#E1E0D9"
SPINE = "#C3C2B7"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titleweight": "bold",
        }
    )


def load_gatv2_matrix(path: Path = GATV2_CSV) -> np.ndarray:
    rows: dict[int, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[int(row["rung"])] = {graph: float(row[graph]) for graph in CANON}
    missing = [rung for rung in RUNGS if int(rung) not in rows]
    if missing:
        raise ValueError(f"incomplete GATv2 matrix; missing rungs {missing}")
    return np.array([[rows[rung][graph] for graph in CANON] for rung in RUNGS])


def load_comparison(path: Path = COMPARISON_CSV) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "rung": int(row["rung"]),
                    "test_graph": row["test_graph"],
                    "entry_rung": int(row["entry_rung"]),
                    "in_training_merge": bool(int(row["in_training_merge"])),
                    "sage_auc": float(row["sage_auc"]),
                    "gatv2_auc": float(row["gatv2_auc"]),
                }
            )
    if len(rows) != 64:
        raise ValueError(f"expected 64 paired cells, found {len(rows)}")
    return rows


def entry_deltas(matrix: np.ndarray) -> np.ndarray:
    """Return entry deltas for rungs 2..8 in canonical graph order."""
    return np.array([matrix[rung - 1, rung - 1] - matrix[rung - 2, rung - 1] for rung in range(2, 9)])


def sage_matrix(rows: list[dict[str, object]]) -> np.ndarray:
    lookup = {(int(row["rung"]), str(row["test_graph"])): float(row["sage_auc"]) for row in rows}
    return np.array([[lookup[(rung, graph)] for graph in CANON] for rung in range(1, 9)])


def declutter(items: list[dict[str, object]], gap: float, lo: float, hi: float) -> list[dict[str, object]]:
    ordered = sorted(items, key=lambda item: float(item["y_true"]))
    for item in ordered:
        item["y_label"] = float(item["y_true"])
    for index in range(1, len(ordered)):
        previous = float(ordered[index - 1]["y_label"])
        current = float(ordered[index]["y_label"])
        if current - previous < gap:
            ordered[index]["y_label"] = previous + gap
    if float(ordered[-1]["y_label"]) > hi:
        ordered[-1]["y_label"] = hi
        for index in range(len(ordered) - 2, -1, -1):
            above = float(ordered[index + 1]["y_label"])
            current = float(ordered[index]["y_label"])
            if above - current < gap:
                ordered[index]["y_label"] = above - gap
    for item in ordered:
        item["y_label"] = min(max(float(item["y_label"]), lo), hi)
    return ordered


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        output = FIGURES / f"{stem}.{extension}"
        fig.savefig(output, bbox_inches="tight", dpi=240)
        print(f"wrote {output}")


def plot_trajectory(matrix: np.ndarray) -> None:
    configure_style()
    x = np.arange(8)
    fig, ax = plt.subplots(figsize=(10.8, 5.8), dpi=200)
    right_labels: list[dict[str, object]] = []
    deltas = entry_deltas(matrix)

    for column, (_, label, entry) in enumerate(GRAPHS):
        values = matrix[:, column]
        entry_index = entry - 1
        for index in range(7):
            in_merge = index + 1 >= entry_index
            ax.plot(
                x[index : index + 2],
                values[index : index + 2],
                color=BLUE if in_merge else GRAY,
                lw=1.9 if in_merge else 1.35,
                ls="-" if in_merge else (0, (4, 2)),
                solid_capstyle="round",
                zorder=3,
            )
        for index, value in enumerate(values):
            in_merge = index >= entry_index
            is_entry = index == entry_index and entry > 1
            ax.scatter(
                index,
                value,
                s=82 if is_entry else 22,
                facecolor=BLUE if in_merge else "white",
                edgecolor="white" if is_entry else (BLUE if in_merge else GRAY),
                linewidth=1.5 if is_entry else 1.2,
                zorder=6,
            )
        if entry >= 4:
            delta = deltas[entry - 2]
            ax.annotate(
                f"+{delta:.3f}",
                xy=(entry_index, values[entry_index]),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.2,
                color=BLUE_DARK,
                fontweight="bold",
            )
        right_labels.append({"y_true": float(values[-1]), "label": label})

    mean = matrix.mean(axis=1)
    ax.plot(
        x,
        mean,
        color=INK,
        lw=2.7,
        marker="s",
        ms=5.7,
        markerfacecolor=INK,
        markeredgecolor="white",
        markeredgewidth=1.0,
        zorder=8,
    )
    ax.annotate(
        "mean (all 8)",
        xy=(x[0] - 0.05, mean[0]),
        ha="right",
        va="center",
        fontsize=8.8,
        color=INK,
        fontweight="bold",
    )

    for item in declutter(right_labels, gap=0.018, lo=0.714, hi=0.991):
        y_true = float(item["y_true"])
        y_label = float(item["y_label"])
        ax.plot([7.06, 7.27], [y_true, y_label], color=MUTED, lw=0.6, zorder=2)
        ax.text(7.33, y_label, str(item["label"]), ha="left", va="center", fontsize=8.7, color=BLUE_DARK)

    ax.set_xlim(-0.58, 8.35)
    ax.set_ylim(0.70, 1.0)
    ax.set_xticks(x, XTICKS)
    ax.set_yticks(np.arange(0.70, 1.001, 0.05))
    ax.set_xlabel("SSL pre-training mixture (one source added per rung)", fontsize=10.3, color=INK)
    ax.set_ylabel("NM AUROC (3-shot, 30-way)", fontsize=10.3, color=INK)
    ax.set_title("GATv2 reproduces the entry-aligned graph ladder", loc="left", fontsize=13, pad=24)
    ax.text(
        0,
        1.02,
        "matched 40k checkpoints · fixed paired episodes · labels mark primary entry jumps",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=MUTED,
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color=BLUE, lw=1.9, marker="o", markerfacecolor=BLUE, markeredgecolor="white", label="in training mixture"),
            Line2D([0], [0], color=GRAY, lw=1.35, ls=(0, (4, 2)), marker="o", markerfacecolor="white", markeredgecolor=GRAY, label="held out"),
            Line2D([0], [0], color=INK, lw=2.7, marker="s", markerfacecolor=INK, markeredgecolor="white", label="mean over all graphs"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=8.7,
        handlelength=2.3,
    )
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    fig.tight_layout()
    save_figure(fig, "nm_ladder_gatv2_trajectory")
    plt.close(fig)


def plot_backbone_comparison(gatv2: np.ndarray, comparison: list[dict[str, object]]) -> None:
    configure_style()
    sage = sage_matrix(comparison)
    gat_deltas = entry_deltas(gatv2)
    sage_deltas = entry_deltas(sage)
    labels = [label for _, label, _ in GRAPHS[1:]]

    fig, (ax_delta, ax_parity) = plt.subplots(1, 2, figsize=(11.4, 4.8), dpi=200, constrained_layout=True)

    y = np.arange(len(labels))
    for index in range(len(labels)):
        ax_delta.plot([sage_deltas[index], gat_deltas[index]], [index, index], color=SPINE, lw=1.4, zorder=1)
    ax_delta.scatter(sage_deltas, y, s=48, facecolor="white", edgecolor=INK, linewidth=1.4, marker="o", label="GraphSAGE", zorder=3)
    ax_delta.scatter(gat_deltas, y, s=52, facecolor=BLUE, edgecolor="white", linewidth=0.8, marker="s", label="GATv2", zorder=4)
    for index, value in enumerate(gat_deltas):
        label_x = max(value, sage_deltas[index]) + 0.005
        ax_delta.text(label_x, index, f"{value:+.3f}", ha="left", va="center", fontsize=8.1, color=BLUE_DARK)
    ax_delta.axvline(0, color=SPINE, lw=0.9)
    ax_delta.axvline(0.02, color=CORAL, lw=1.0, ls=(0, (3, 2)))
    ax_delta.text(0.0225, -0.65, ".02 reference", color=CORAL, fontsize=7.8, ha="left", va="center")
    ax_delta.set_yticks(y, labels)
    ax_delta.invert_yaxis()
    ax_delta.set_xlim(-0.005, 0.195)
    ax_delta.set_xticks(np.arange(0, 0.201, 0.05))
    ax_delta.set_xlabel("AUROC change when source enters")
    ax_delta.set_title("Entry jumps agree across backbones", loc="left", fontsize=11.5)
    ax_delta.legend(loc="lower right", frameon=False, fontsize=8.4)
    ax_delta.grid(axis="x", color=GRID, lw=0.75)
    ax_delta.set_axisbelow(True)

    sage_auc = np.array([float(row["sage_auc"]) for row in comparison])
    gat_auc = np.array([float(row["gatv2_auc"]) for row in comparison])
    in_merge = np.array([bool(row["in_training_merge"]) for row in comparison])
    ax_parity.plot([0.70, 0.99], [0.70, 0.99], color=INK, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax_parity.scatter(
        sage_auc[~in_merge],
        gat_auc[~in_merge],
        s=34,
        facecolor="white",
        edgecolor=GRAY,
        linewidth=1.1,
        marker="o",
        label="held out",
        zorder=2,
    )
    ax_parity.scatter(
        sage_auc[in_merge],
        gat_auc[in_merge],
        s=36,
        facecolor=BLUE,
        edgecolor="white",
        linewidth=0.7,
        marker="s",
        label="in mixture",
        zorder=3,
    )
    correlation = float(np.corrcoef(sage_auc, gat_auc)[0, 1])
    mean_gap = float(np.mean(gat_auc - sage_auc))
    ax_parity.text(
        0.03,
        0.97,
        f"r = {correlation:.3f}\nmean GATv2 − SAGE = {mean_gap:+.3f}",
        transform=ax_parity.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=INK,
    )
    ax_parity.set_xlim(0.70, 0.99)
    ax_parity.set_ylim(0.70, 0.99)
    ax_parity.set_aspect("equal", adjustable="box")
    ax_parity.set_xlabel("GraphSAGE AUROC")
    ax_parity.set_ylabel("GATv2 AUROC")
    ax_parity.set_title("The complete 8×8 matrices are nearly identical", loc="left", fontsize=11.5)
    ax_parity.legend(loc="lower right", frameon=False, fontsize=8.4)
    ax_parity.grid(color=GRID, lw=0.7)
    ax_parity.set_axisbelow(True)

    for axis in (ax_delta, ax_parity):
        axis.tick_params(colors=MUTED, labelsize=8.5)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(SPINE)

    fig.suptitle("GATv2 preserves the graph-ladder mechanism", fontsize=13.2, fontweight="bold")
    save_figure(fig, "nm_ladder_gatv2_backbone_comparison")
    plt.close(fig)

    print("entry delta (GraphSAGE, GATv2):")
    for label, sage_delta, gat_delta in zip(labels, sage_deltas, gat_deltas):
        print(f"  {label:14s} {sage_delta:+.4f}  {gat_delta:+.4f}")
    print(f"all-cell correlation: {correlation:.6f}")
    print(f"mean GATv2 - GraphSAGE: {mean_gap:+.6f}")


def main() -> None:
    gatv2 = load_gatv2_matrix()
    comparison = load_comparison()
    plot_trajectory(gatv2)
    plot_backbone_comparison(gatv2, comparison)


if __name__ == "__main__":
    main()
