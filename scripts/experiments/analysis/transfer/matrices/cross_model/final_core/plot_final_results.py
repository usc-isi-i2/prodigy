#!/usr/bin/env python3
"""Create the canonical final-core matrix, ladder, and coverage figures.

All figures are derived only from ``data/results_full_long.tsv``. Run locally:

    /opt/homebrew/bin/python3.11 \
      scripts/experiments/analysis/transfer/matrices/cross_model/final_core/plot_final_results.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "data/results_full_long.tsv"
OUT = HERE / "figures"

GRAPHS = (
    "ukr_rus",
    "covid",
    "midterm",
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
    "cp_hk",
    "facebook_page_reference",
)
SHORT = {
    "ukr_rus": "UKR/RUS",
    "covid": "COVID-19",
    "midterm": "Midterm",
    "covid_political": "COVID-pol.",
    "election2020": "Election ’20",
    "ukr_rus_suspended": "UKR/RUS susp.",
    "twibot20": "TwiBot-20",
    "cp_hk": "CP-HK",
    "facebook_page_reference": "Facebook",
}
ORDERS = ("A", "B", "C")

BLUE = "#2878b5"
ORANGE = "#d56a32"
GREEN = "#28866d"
PURPLE = "#7a5aa6"
RED = "#bd3f45"
INK = "#171717"
MUTED = "#6f6f6f"
GRID = "#dddddd"
LIGHT = "#eeeeee"
ORDER_COLORS = {"A": BLUE, "B": ORANGE, "C": GREEN}
TARGET_COLORS = dict(zip(GRAPHS, plt.get_cmap("tab10").colors[:9]))

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def load_rows() -> list[dict[str, str]]:
    with DATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 1944 or len({row["cell_id"] for row in rows}) != 1944:
        raise ValueError("canonical table is not the exact 1,944-cell design")
    return rows


def observed(rows: list[dict[str, str]], architecture: str, component: str) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["architecture"] == architecture
        and row["component"] == component
        and row["result_status"] == "observed"
    ]


def clean_axis(ax: Any, *, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bbbbbb")
    ax.spines["bottom"].set_color("#bbbbbb")
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.65, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.10,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=INK,
    )


def save(fig: Any, stem: str) -> None:
    png_out = OUT / "pngs"
    pdf_out = OUT / "pdfs"
    png_out.mkdir(parents=True, exist_ok=True)
    pdf_out.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_out / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(
        pdf_out / f"{stem}.pdf",
        dpi=240,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def matrix_values(rows: list[dict[str, str]], architecture: str) -> np.ndarray:
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    metric = "nm_roc_auc_ovr_macro" if architecture == "PRODIGY" else "graphcl_loss"
    for row in observed(rows, architecture, "matrix"):
        train_graphs = json.loads(row["train_graphs"])
        if len(train_graphs) != 1:
            raise ValueError(f"matrix row has {len(train_graphs)} training graphs")
        buckets[(train_graphs[0], row["test_graph"])].append(float(row[metric]))
    expected_replicates = 3
    result = np.zeros((len(GRAPHS), len(GRAPHS)))
    for i, source in enumerate(GRAPHS):
        for j, target in enumerate(GRAPHS):
            values = buckets[(source, target)]
            if len(values) != expected_replicates:
                raise ValueError(f"{architecture} matrix cell {(source, target)} has {len(values)} replicates")
            result[i, j] = float(np.mean(values))
    return result


def plot_specialist_matrices(rows: list[dict[str, str]]) -> None:
    prodigy = matrix_values(rows, "PRODIGY")
    samgpt_loss = matrix_values(rows, "SAMGPT")
    samgpt_score = -np.log10(np.maximum(samgpt_loss, np.finfo(float).tiny))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.0), constrained_layout=True)
    panels = (
        (axes[0], prodigy, "PRODIGY · neighbor matching", "ROC-AUC (mean of 3 seeds)", "YlGnBu", "A"),
        (axes[1], samgpt_score, "SAMGPT · GraphCL", "−log₁₀(BCE loss), mean of 3 seeds", "YlOrRd", "B"),
    )
    for ax, values, title, colorbar_label, cmap, label in panels:
        image = ax.imshow(values, cmap=cmap, aspect="equal")
        for index in range(len(GRAPHS)):
            ax.add_patch(Rectangle((index - 0.5, index - 0.5), 1, 1, fill=False, edgecolor=INK, lw=1.25))
        ax.set_xticks(range(len(GRAPHS)), [SHORT[g] for g in GRAPHS], rotation=42, ha="right")
        ax.set_yticks(range(len(GRAPHS)), [SHORT[g] for g in GRAPHS])
        ax.set_xlabel("evaluation target")
        ax.set_ylabel("specialist training source")
        ax.set_title(title, loc="left", fontweight="bold", pad=12)
        panel_label(ax, label)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        colorbar.set_label(colorbar_label)
        for spine in ax.spines.values():
            spine.set_visible(False)
    save(fig, "specialist_transfer_matrices")


def entry_events(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    ladder = [row for row in rows if row["component"] == "ladder" and row["result_status"] == "observed"]
    index = {
        (
            row["architecture"],
            row["training_seed_slot"],
            row["order"],
            int(row["rung"]),
            row["test_graph"],
        ): row
        for row in ladder
    }
    events = []
    for row in ladder:
        rung = int(row["rung"])
        if rung == 1 or row["test_graph"] != row["added_graph"]:
            continue
        key = (
            row["architecture"],
            row["training_seed_slot"],
            row["order"],
            rung - 1,
            row["test_graph"],
        )
        before = float(index[key]["primary_value"])
        after = float(row["primary_value"])
        effect = after - before if row["architecture"] == "PRODIGY" else before - after
        events.append(
            {
                "architecture": row["architecture"],
                "seed_slot": int(row["training_seed_slot"]),
                "order": row["order"],
                "target": row["test_graph"],
                "rung": rung,
                "before": before,
                "after": after,
                "effect": effect,
            }
        )
    counts = {architecture: sum(event["architecture"] == architecture for event in events) for architecture in ("PRODIGY", "SAMGPT")}
    positives = {architecture: sum(event["architecture"] == architecture and event["effect"] > 0 for event in events) for architecture in counts}
    if counts != {"PRODIGY": 72, "SAMGPT": 72} or positives != {"PRODIGY": 72, "SAMGPT": 49}:
        raise ValueError(f"entry-event contract changed: counts={counts}, positives={positives}")
    return events


def deterministic_jitter(index: int, total: int, width: float = 0.22) -> float:
    if total <= 1:
        return 0.0
    return -width + 2 * width * index / (total - 1)


def plot_entry_effects(events: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.7), constrained_layout=True)
    configs = (
        ("PRODIGY", axes[0], "accuracy gain (after − before)", "72/72 improve", "A"),
        ("SAMGPT", axes[1], "BCE reduction (before − after)", "49/72 improve", "B"),
    )
    for architecture, ax, xlabel, count_label, label in configs:
        subset = [event for event in events if event["architecture"] == architecture]
        for y, target in enumerate(GRAPHS):
            target_events = [event for event in subset if event["target"] == target]
            target_events.sort(key=lambda event: (event["order"], event["seed_slot"]))
            values = np.array([event["effect"] for event in target_events])
            if len(values):
                ax.plot([values.min(), values.max()], [y, y], color="#b8b8b8", lw=1.0, zorder=1)
                for point_index, event in enumerate(target_events):
                    ax.scatter(
                        event["effect"],
                        y + deterministic_jitter(point_index, len(target_events)),
                        s=27,
                        color=ORDER_COLORS[event["order"]],
                        alpha=0.80,
                        edgecolor="white",
                        linewidth=0.45,
                        zorder=3,
                    )
                ax.scatter(float(np.mean(values)), y, marker="D", s=31, color=INK, edgecolor="white", linewidth=0.5, zorder=5)
        ax.axvline(0, color=RED, lw=1.0, ls=(0, (3, 2)))
        ax.set_yticks(range(len(GRAPHS)), [SHORT[g] for g in GRAPHS])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel + "  ·  positive is better")
        ax.set_title(f"{architecture} target-entry effects", loc="left", fontweight="bold", pad=12)
        ax.text(0.99, 0.98, count_label, transform=ax.transAxes, ha="right", va="top", color=INK, fontweight="bold")
        if architecture == "SAMGPT":
            ax.set_xscale("symlog", linthresh=5e-4)
            ax.set_xticks([-1e-3, 0, 1e-3, 1e-2, 1e-1])
            ax.set_xlabel(xlabel + "  ·  positive is better  ·  symmetric log scale")
        clean_axis(ax, grid_axis="x")
        panel_label(ax, label)
    axes[1].legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=ORDER_COLORS[order], markeredgecolor="white", label=f"order {order}")
            for order in ORDERS
        ]
        + [Line2D([0], [0], marker="D", color="none", markerfacecolor=INK, markeredgecolor="white", label="target mean")],
        loc="lower right",
        frameon=False,
    )
    save(fig, "target_entry_effects")


def plot_before_after(events: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 5.2), constrained_layout=True)
    for architecture, ax, label in (("PRODIGY", axes[0], "A"), ("SAMGPT", axes[1], "B")):
        subset = [event for event in events if event["architecture"] == architecture]
        before = np.array([event["before"] for event in subset])
        after = np.array([event["after"] for event in subset])
        lower = min(before.min(), after.min())
        upper = max(before.max(), after.max())
        if architecture == "SAMGPT":
            lower *= 0.72
            upper *= 1.35
            ax.set_xscale("log")
            ax.set_yscale("log")
        else:
            pad = (upper - lower) * 0.08
            lower -= pad
            upper += pad
        ax.plot([lower, upper], [lower, upper], color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
        for order in ORDERS:
            order_events = [event for event in subset if event["order"] == order]
            ax.scatter(
                [event["before"] for event in order_events],
                [event["after"] for event in order_events],
                s=35,
                color=ORDER_COLORS[order],
                alpha=0.78,
                edgecolor="white",
                linewidth=0.55,
                label=f"order {order}",
                zorder=3,
            )
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("before target enters")
        ax.set_ylabel("after target enters")
        direction = "above diagonal improves" if architecture == "PRODIGY" else "below diagonal improves"
        ax.set_title(f"{architecture} · {direction}", loc="left", fontweight="bold", pad=12)
        clean_axis(ax, grid_axis="both")
        panel_label(ax, label)
    axes[1].legend(loc="upper left", frameon=False)
    save(fig, "target_entry_before_after")


def scatter_box(ax: Any, groups: list[list[float]], labels: list[str], colors: list[str]) -> None:
    positions = np.arange(1, len(groups) + 1)
    box = ax.boxplot(
        groups,
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": INK, "lw": 1.4},
        whiskerprops={"color": MUTED},
        capprops={"color": MUTED},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.16)
        patch.set_edgecolor(color)
    for group_index, (values, color) in enumerate(zip(groups, colors), 1):
        for point_index, value in enumerate(values):
            jitter = deterministic_jitter(point_index % 9, min(len(values), 9), 0.16)
            ax.scatter(group_index + jitter, value, s=20, color=color, alpha=0.70, edgecolor="white", linewidth=0.35, zorder=3)
        ax.scatter(group_index, float(np.mean(values)), marker="D", s=38, color=INK, edgecolor="white", linewidth=0.5, zorder=5)
    ax.set_xticks(positions, labels)


def plot_order_robustness(events: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    for architecture, ax, ylabel, label in (
        ("PRODIGY", axes[0], "accuracy gain", "A"),
        ("SAMGPT", axes[1], "BCE reduction", "B"),
    ):
        groups = [
            [event["effect"] for event in events if event["architecture"] == architecture and event["order"] == order]
            for order in ORDERS
        ]
        scatter_box(ax, groups, [f"Order {order}\n(n={len(group)})" for order, group in zip(ORDERS, groups)], [ORDER_COLORS[o] for o in ORDERS])
        ax.axhline(0, color=RED, lw=1.0, ls=(0, (3, 2)))
        ax.set_ylabel(ylabel + "  ·  positive is better")
        ax.set_title(f"{architecture} order robustness", loc="left", fontweight="bold", pad=12)
        if architecture == "SAMGPT":
            ax.set_yscale("symlog", linthresh=5e-4)
            ax.set_yticks([-1e-3, 0, 1e-3, 1e-2, 1e-1])
            ax.set_ylabel(ylabel + "  ·  positive is better  ·  symmetric log scale")
        clean_axis(ax)
        panel_label(ax, label)
    save(fig, "order_robustness")


def plot_seed_stability(events: list[dict[str, Any]]) -> None:
    prodigy = [event for event in events if event["architecture"] == "PRODIGY"]
    groups = [[event["effect"] for event in prodigy if event["seed_slot"] == seed] for seed in (0, 1, 2)]
    fig, ax = plt.subplots(figsize=(6.7, 4.5), constrained_layout=True)
    scatter_box(ax, groups, [f"Seed {seed}\n(n={len(group)})" for seed, group in zip((0, 1, 2), groups)], [BLUE, PURPLE, GREEN])
    ax.axhline(0, color=RED, lw=1.0, ls=(0, (3, 2)))
    ax.set_ylabel("target-entry accuracy gain  ·  positive is better")
    ax.set_title("PRODIGY entry gains are stable across training seeds", loc="left", fontweight="bold", pad=12)
    clean_axis(ax)
    means = [float(np.mean(group)) for group in groups]
    ax.text(0.99, 0.98, "means: " + " · ".join(f"{value:+.3f}" for value in means), transform=ax.transAxes, ha="right", va="top", color=MUTED)
    save(fig, "prodigy_seed_stability")


def ladder_series(
    rows: list[dict[str, str]],
    architecture: str,
    metric: str = "primary_value",
) -> dict[tuple[str, str], list[float]]:
    buckets: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in observed(rows, architecture, "ladder"):
        buckets[(row["order"], row["test_graph"], int(row["rung"]))].append(float(row[metric]))
    expected_replicates = 3
    result = {}
    for order in ORDERS:
        for target in GRAPHS:
            values = []
            for rung in range(1, 10):
                replicates = buckets[(order, target, rung)]
                if len(replicates) != expected_replicates:
                    raise ValueError(f"{architecture} ladder cell {(order, target, rung)} has {len(replicates)} replicates")
                values.append(float(np.mean(replicates)))
            result[(order, target)] = values
    return result


def entry_rungs(rows: list[dict[str, str]], architecture: str) -> dict[tuple[str, str], int]:
    result = {}
    for row in observed(rows, architecture, "ladder"):
        if row["test_graph"] == row["added_graph"]:
            result[(row["order"], row["test_graph"])] = int(row["rung"])
    if len(result) != 27:
        raise ValueError(f"{architecture} has {len(result)} entry-rung mappings, expected 27")
    return result


def plot_ladder_trajectories(rows: list[dict[str, str]]) -> None:
    series = {architecture: ladder_series(rows, architecture) for architecture in ("PRODIGY", "SAMGPT")}
    entries = {architecture: entry_rungs(rows, architecture) for architecture in ("PRODIGY", "SAMGPT")}
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharex=True)
    for row_index, architecture in enumerate(("PRODIGY", "SAMGPT")):
        for column_index, order in enumerate(ORDERS):
            ax = axes[row_index, column_index]
            mean_by_rung = []
            for target in GRAPHS:
                values = series[architecture][(order, target)]
                rungs = np.arange(1, 10)
                ax.plot(rungs, values, color=TARGET_COLORS[target], lw=1.25, alpha=0.78)
                entry = entries[architecture][(order, target)]
                ax.scatter(entry, values[entry - 1], s=31, color=TARGET_COLORS[target], edgecolor="white", linewidth=0.55, zorder=4)
            for rung in range(9):
                mean_by_rung.append(float(np.mean([series[architecture][(order, target)][rung] for target in GRAPHS])))
            ax.plot(range(1, 10), mean_by_rung, color=INK, lw=2.3, marker="s", ms=3.8, zorder=5)
            if architecture == "SAMGPT":
                ax.set_yscale("log")
            ax.set_title(f"Order {order}", fontweight="bold")
            ax.set_xticks(range(1, 10))
            ax.set_xlabel("cumulative training rung")
            if column_index == 0:
                ylabel = "NM accuracy\n(mean of 3 seeds)" if architecture == "PRODIGY" else "GraphCL BCE loss\n(mean of 3 seeds; log scale)"
                ax.set_ylabel(ylabel)
            clean_axis(ax)
            if column_index == 0:
                panel_label(ax, "A" if architecture == "PRODIGY" else "B")
        axes[row_index, 0].text(-0.22, 0.5, architecture, transform=axes[row_index, 0].transAxes, rotation=90, ha="center", va="center", fontsize=11, fontweight="bold")
    handles = [Line2D([0], [0], color=TARGET_COLORS[target], lw=2, label=SHORT[target]) for target in GRAPHS]
    handles.append(Line2D([0], [0], color=INK, marker="s", lw=2.3, label="mean over targets"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0.03, 0.10, 1.0, 1.0), h_pad=2.0, w_pad=1.7)
    save(fig, "ladder_trajectories")


def plot_ladder_loss_trajectories(rows: list[dict[str, str]]) -> None:
    """Plot both architectures' native-pretext losses over the ladder."""
    metrics = {"PRODIGY": "nm_loss", "SAMGPT": "graphcl_loss"}
    series = {
        architecture: ladder_series(rows, architecture, metric)
        for architecture, metric in metrics.items()
    }
    entries = {architecture: entry_rungs(rows, architecture) for architecture in metrics}
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharex=True)
    rungs = np.arange(1, 10)
    for row_index, architecture in enumerate(("PRODIGY", "SAMGPT")):
        for column_index, order in enumerate(ORDERS):
            ax = axes[row_index, column_index]
            for target in GRAPHS:
                values = series[architecture][(order, target)]
                ax.plot(rungs, values, color=TARGET_COLORS[target], lw=1.25, alpha=0.78)
                entry = entries[architecture][(order, target)]
                ax.scatter(
                    entry,
                    values[entry - 1],
                    s=31,
                    color=TARGET_COLORS[target],
                    edgecolor="white",
                    linewidth=0.55,
                    zorder=4,
                )
            mean_by_rung = [
                float(np.mean([series[architecture][(order, target)][rung - 1] for target in GRAPHS]))
                for rung in range(1, 10)
            ]
            ax.plot(rungs, mean_by_rung, color=INK, lw=2.3, marker="s", ms=3.8, zorder=5)
            if architecture == "SAMGPT":
                ax.set_yscale("log")
            ax.set_title(f"Order {order}", fontweight="bold")
            ax.set_xticks(range(1, 10))
            ax.set_xlabel("cumulative training rung")
            if column_index == 0:
                ylabel = "NM loss\n(mean of 3 seeds)" if architecture == "PRODIGY" else "GraphCL BCE loss\n(mean of 3 seeds; log scale)"
                ax.set_ylabel(ylabel)
                panel_label(ax, "A" if architecture == "PRODIGY" else "B")
            clean_axis(ax)
        axes[row_index, 0].text(
            -0.22,
            0.5,
            architecture,
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
    handles = [Line2D([0], [0], color=TARGET_COLORS[target], lw=2, label=SHORT[target]) for target in GRAPHS]
    handles.append(Line2D([0], [0], color=INK, marker="s", lw=2.3, label="mean over targets"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0.03, 0.10, 1.0, 1.0), h_pad=2.0, w_pad=1.7)
    save(fig, "ladder_trajectories_loss")


def plot_ladder_native_accuracy(rows: list[dict[str, str]]) -> None:
    """Plot each architecture's native-pretext evaluation accuracy."""
    metrics = {"PRODIGY": "nm_accuracy", "SAMGPT": "graphcl_accuracy"}
    series = {
        architecture: ladder_series(rows, architecture, metric)
        for architecture, metric in metrics.items()
    }
    entries = {architecture: entry_rungs(rows, architecture) for architecture in metrics}
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharex=True)
    rungs = np.arange(1, 10)
    for row_index, architecture in enumerate(("PRODIGY", "SAMGPT")):
        for column_index, order in enumerate(ORDERS):
            ax = axes[row_index, column_index]
            for target in GRAPHS:
                values = series[architecture][(order, target)]
                ax.plot(rungs, values, color=TARGET_COLORS[target], lw=1.25, alpha=0.78)
                entry = entries[architecture][(order, target)]
                ax.scatter(entry, values[entry - 1], s=31, color=TARGET_COLORS[target], edgecolor="white", linewidth=0.55, zorder=4)
            mean_by_rung = [
                float(np.mean([series[architecture][(order, target)][rung - 1] for target in GRAPHS]))
                for rung in range(1, 10)
            ]
            ax.plot(rungs, mean_by_rung, color=INK, lw=2.3, marker="s", ms=3.8, zorder=5)
            ax.set_title(f"Order {order}", fontweight="bold")
            ax.set_xticks(range(1, 10))
            ax.set_xlabel("cumulative training rung")
            if architecture == "SAMGPT":
                ax.set_ylim(0.84, 1.005)
            if column_index == 0:
                ylabel = "NM accuracy\n(mean of 3 seeds)" if architecture == "PRODIGY" else "GraphCL discrimination accuracy\n(mean of 3 seeds)"
                ax.set_ylabel(ylabel)
                panel_label(ax, "A" if architecture == "PRODIGY" else "B")
            clean_axis(ax)
        axes[row_index, 0].text(-0.22, 0.5, architecture, transform=axes[row_index, 0].transAxes, rotation=90, ha="center", va="center", fontsize=11, fontweight="bold")
    handles = [Line2D([0], [0], color=TARGET_COLORS[target], lw=2, label=SHORT[target]) for target in GRAPHS]
    handles.append(Line2D([0], [0], color=INK, marker="s", lw=2.3, label="mean over targets"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0.03, 0.10, 1.0, 1.0), h_pad=2.0, w_pad=1.7)
    save(fig, "ladder_trajectories_native_accuracy")


def plot_samgpt_probability_diagnostics(rows: list[dict[str, str]]) -> None:
    """Plot SAMGPT GraphCL pair probabilities and their separation margin."""
    panels = (
        ("graphcl_positive_probability", "positive-pair probability"),
        ("graphcl_negative_probability", "negative-pair probability"),
        ("graphcl_probability_margin", "probability margin (positive − negative)"),
    )
    series = {metric: ladder_series(rows, "SAMGPT", metric) for metric, _ in panels}
    entries = entry_rungs(rows, "SAMGPT")
    fig, axes = plt.subplots(3, 3, figsize=(14.0, 11.0), sharex=True)
    rungs = np.arange(1, 10)
    for row_index, (metric, label) in enumerate(panels):
        for column_index, order in enumerate(ORDERS):
            ax = axes[row_index, column_index]
            for target in GRAPHS:
                values = series[metric][(order, target)]
                ax.plot(rungs, values, color=TARGET_COLORS[target], lw=1.2, alpha=0.78)
                entry = entries[(order, target)]
                ax.scatter(entry, values[entry - 1], s=29, color=TARGET_COLORS[target], edgecolor="white", linewidth=0.5, zorder=4)
            mean_by_rung = [
                float(np.mean([series[metric][(order, target)][rung - 1] for target in GRAPHS]))
                for rung in range(1, 10)
            ]
            ax.plot(rungs, mean_by_rung, color=INK, lw=2.3, marker="s", ms=3.6, zorder=5)
            ax.set_title(f"Order {order}", fontweight="bold")
            ax.set_xticks(range(1, 10))
            ax.set_xlabel("cumulative training rung")
            if column_index == 0:
                ax.set_ylabel(label + "\n(mean of 3 seeds)")
                panel_label(ax, chr(ord("A") + row_index))
            clean_axis(ax)
    handles = [Line2D([0], [0], color=TARGET_COLORS[target], lw=2, label=SHORT[target]) for target in GRAPHS]
    handles.append(Line2D([0], [0], color=INK, marker="s", lw=2.3, label="mean over targets"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0.03, 0.075, 1.0, 1.0), h_pad=2.0, w_pad=1.7)
    save(fig, "samgpt_ladder_probability_diagnostics")


def plot_ladder_auc_trajectories(rows: list[dict[str, str]]) -> None:
    """Plot the log-recovered PRODIGY ROC-AUC ladder trajectories."""
    series = ladder_series(rows, "PRODIGY", "nm_roc_auc_ovr_macro")
    entries = entry_rungs(rows, "PRODIGY")
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.7), sharex=True, sharey=True)
    rungs = np.arange(1, 10)
    for column_index, order in enumerate(ORDERS):
        ax = axes[column_index]
        for target in GRAPHS:
            values = series[(order, target)]
            ax.plot(rungs, values, color=TARGET_COLORS[target], lw=1.35, alpha=0.82)
            entry = entries[(order, target)]
            ax.scatter(
                entry,
                values[entry - 1],
                s=31,
                color=TARGET_COLORS[target],
                edgecolor="white",
                linewidth=0.55,
                zorder=4,
            )
        mean_by_rung = [
            float(np.mean([series[(order, target)][rung - 1] for target in GRAPHS]))
            for rung in range(1, 10)
        ]
        ax.plot(rungs, mean_by_rung, color=INK, lw=2.4, marker="s", ms=3.8, zorder=5)
        ax.set_title(f"Order {order}", fontweight="bold")
        ax.set_xticks(range(1, 10))
        ax.set_xlabel("cumulative training rung")
        if column_index == 0:
            ax.set_ylabel("NM ROC-AUC\n(mean of 3 seeds)")
            panel_label(ax, "A")
        clean_axis(ax)

    handles = [Line2D([0], [0], color=TARGET_COLORS[target], lw=2, label=SHORT[target]) for target in GRAPHS]
    handles.append(Line2D([0], [0], color=INK, marker="s", lw=2.3, label="mean over targets"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.055))
    fig.text(
        0.5,
        0.012,
        "PRODIGY · values recovered from original fixed-test logs (four-decimal printed precision)",
        ha="center",
        va="bottom",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0.02, 0.18, 1.0, 1.0), w_pad=1.7)
    save(fig, "ladder_trajectories_auc")


def plot_ladder_seed_bands(rows: list[dict[str, str]]) -> None:
    """Plot PRODIGY ladder means and observed ranges across training seeds."""
    buckets: dict[tuple[int, str, str], dict[int, float]] = defaultdict(dict)
    for row in observed(rows, "PRODIGY", "ladder"):
        key = (int(row["training_seed_slot"]), row["order"], row["test_graph"])
        rung = int(row["rung"])
        if rung in buckets[key]:
            raise ValueError(f"duplicate PRODIGY ladder cell {key + (rung,)}")
        buckets[key][rung] = float(row["primary_value"])

    values: dict[tuple[int, str, str], np.ndarray] = {}
    for seed_slot in range(3):
        for order in ORDERS:
            for target in GRAPHS:
                rung_values = buckets[(seed_slot, order, target)]
                if set(rung_values) != set(range(1, 10)):
                    raise ValueError(
                        f"PRODIGY seed-band series {(seed_slot, order, target)} "
                        f"has rungs {sorted(rung_values)}"
                    )
                values[(seed_slot, order, target)] = np.array(
                    [rung_values[rung] for rung in range(1, 10)]
                )

    entries = entry_rungs(rows, "PRODIGY")
    rungs = np.arange(1, 10)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.7), sharex=True, sharey=True)
    for column_index, order in enumerate(ORDERS):
        ax = axes[column_index]
        seed_target_means = np.zeros((3, 9))
        for target in GRAPHS:
            seed_values = np.vstack(
                [values[(seed_slot, order, target)] for seed_slot in range(3)]
            )
            mean = seed_values.mean(axis=0)
            ax.fill_between(
                rungs,
                seed_values.min(axis=0),
                seed_values.max(axis=0),
                color=TARGET_COLORS[target],
                alpha=0.11,
                linewidth=0,
                zorder=1,
            )
            ax.plot(rungs, mean, color=TARGET_COLORS[target], lw=1.35, alpha=0.88, zorder=2)
            entry = entries[(order, target)]
            ax.scatter(
                entry,
                mean[entry - 1],
                s=31,
                color=TARGET_COLORS[target],
                edgecolor="white",
                linewidth=0.55,
                zorder=4,
            )
            for seed_slot in range(3):
                seed_target_means[seed_slot] += seed_values[seed_slot] / len(GRAPHS)

        overall_mean = seed_target_means.mean(axis=0)
        ax.fill_between(
            rungs,
            seed_target_means.min(axis=0),
            seed_target_means.max(axis=0),
            color=INK,
            alpha=0.14,
            linewidth=0,
            zorder=3,
        )
        ax.plot(rungs, overall_mean, color=INK, lw=2.4, marker="s", ms=3.8, zorder=5)
        ax.set_title(f"Order {order}", fontweight="bold")
        ax.set_xticks(range(1, 10))
        ax.set_xlabel("cumulative training rung")
        if column_index == 0:
            ax.set_ylabel("NM accuracy")
            panel_label(ax, "A")
        clean_axis(ax)

    handles = [Line2D([0], [0], color=TARGET_COLORS[target], lw=2, label=SHORT[target]) for target in GRAPHS]
    handles.extend(
        [
            Line2D([0], [0], color=INK, marker="s", lw=2.3, label="mean over targets"),
            Patch(facecolor="#888888", alpha=0.22, edgecolor="none", label="seed min–max range"),
        ]
    )
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.03))
    fig.text(
        0.5,
        0.005,
        "Lines show means across training seeds 0, 1, and 2 · shaded areas show the observed min–max range (n=3)",
        ha="center",
        va="bottom",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0.02, 0.18, 1.0, 1.0), w_pad=1.7)
    save(fig, "prodigy_ladder_seed_bands")


def plot_coverage(rows: list[dict[str, str]]) -> None:
    row_specs = (
        ("PRODIGY", "matrix", 81),
        ("PRODIGY", "ladder", 243),
        ("SAMGPT", "matrix", 81),
        ("SAMGPT", "ladder", 243),
    )
    fig, ax = plt.subplots(figsize=(7.9, 3.8), constrained_layout=True)
    for y, (architecture, component, per_seed) in enumerate(row_specs):
        for seed_slot in range(3):
            subset = [
                row
                for row in rows
                if row["architecture"] == architecture
                and row["component"] == component
                and int(row["training_seed_slot"]) == seed_slot
            ]
            if len(subset) != per_seed:
                raise ValueError(f"coverage block {(architecture, component, seed_slot)} has {len(subset)} rows")
            is_complete = all(row["result_status"] == "observed" for row in subset)
            color = GREEN if is_complete else "#cfcfcf"
            ax.add_patch(Rectangle((seed_slot - 0.42, y - 0.36), 0.84, 0.72, facecolor=color, edgecolor="white", lw=1.5))
            seed_text = subset[0]["training_seed"] or "pending"
            ax.text(seed_slot, y - 0.06, f"{per_seed} cells", ha="center", va="center", color="white" if is_complete else INK, fontweight="bold")
            ax.text(seed_slot, y + 0.18, f"seed {seed_text}" if seed_text != "pending" else seed_text, ha="center", va="center", color="white" if is_complete else MUTED, fontsize=7.5)
        observed_count = sum(
            row["result_status"] == "observed"
            for row in rows
            if row["architecture"] == architecture and row["component"] == component
        )
        total_count = per_seed * 3
        ax.text(3.05, y, f"{observed_count}/{total_count}", ha="left", va="center", color=INK, fontweight="bold")
    ax.set_xlim(-0.55, 3.75)
    ax.set_ylim(-0.65, len(row_specs) - 0.35)
    ax.invert_yaxis()
    ax.set_xticks(range(3), ["Seed slot 1", "Seed slot 2", "Seed slot 3"])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(row_specs)), [f"{architecture}\n{component}" for architecture, component, _ in row_specs])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(3.05, -0.62, "observed / planned", ha="left", va="bottom", fontsize=8, color=MUTED)
    ax.legend(
        handles=[Patch(facecolor=GREEN, label="observed"), Patch(facecolor="#cfcfcf", label="pending")],
        loc="lower right",
        frameon=False,
    )
    ax.set_title("Final experiment coverage", loc="left", fontweight="bold", pad=30)
    save(fig, "coverage_status")


def main() -> None:
    rows = load_rows()
    events = entry_events(rows)
    plot_specialist_matrices(rows)
    plot_entry_effects(events)
    plot_before_after(events)
    plot_order_robustness(events)
    plot_seed_stability(events)
    plot_ladder_trajectories(rows)
    plot_ladder_loss_trajectories(rows)
    plot_ladder_native_accuracy(rows)
    plot_samgpt_probability_diagnostics(rows)
    plot_ladder_auc_trajectories(rows)
    plot_ladder_seed_bands(rows)
    plot_coverage(rows)
    print(f"FINAL_CORE_FIGURES_OK figures=12 formats=png,pdf output={OUT}")


if __name__ == "__main__":
    main()
