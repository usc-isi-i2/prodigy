#!/usr/bin/env python3
"""Plot one full native-GraphCL target trajectory for each SAMGPT source order.

Each target is evaluated at every rung. Gray/dashed segments are before the target
enters the cumulative training mixture; blue/solid segments are at and after entry.
The black line is the mean over the same fixed set of nine targets at every rung.

Probability margin is the default because it is bounded and does not let one
catastrophic BCE value dominate an arithmetic mean.  The original BCE figures
remain reproducible with ``--metric loss``.

Run locally with ``/opt/homebrew/bin/python3.11 plot_samgpt_graphcl_native_ladder.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "data" / "samgpt_graphcl_9x3_carc_v100"
FIG_ROOT = HERE / "figures"

BLUE = "#2a78d6"
GRAY = "#8f8d87"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

METRICS = {
    "accuracy": {
        "ylabel": "native GraphCL edge-classification accuracy  (higher is better)",
        "stem": "samgpt_graphcl_native_accuracy_order_{order}",
    },
    "probability_margin": {
        "ylabel": "native GraphCL probability margin  (higher is better)",
        "stem": "samgpt_graphcl_native_probability_margin_order_{order}",
    },
    "loss": {
        "ylabel": "native GraphCL BCE loss  (lower is better)",
        "stem": "samgpt_graphcl_native_ladder_order_{order}",
    },
}

SHORT = {
    "ukr_rus_twitter": "Ukr-Rus",
    "covid19_twitter": "COVID-19",
    "midterm": "Midterm",
    "covid_political": "COVID-pol.",
    "election2020": "Election '20",
    "ukr_rus_suspended": "Ukr-Rus susp.",
    "twibot20": "TwiBot-20",
    "cp_hk_twitter": "CP-HK",
    "facebook_page_reference": "Facebook",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def load(metric: str) -> tuple[dict[str, list[str]], list[str], dict[tuple[str, int], dict]]:
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text())
    orders = {str(key): list(value) for key, value in manifest["orders"].items()}
    targets = list(manifest["targets"])
    table = {}
    values_by_rung: dict[tuple[str, int], dict[str, float]] = {}
    with (DATA_ROOT / "metrics_long.csv").open(newline="") as handle:
        for raw in csv.DictReader(handle):
            key = (raw["order"], int(raw["rung"]))
            values_by_rung.setdefault(key, {})[raw["target"]] = float(raw[metric])
    for key, target_values in values_by_rung.items():
        table[key] = {
            "target_mean": float(np.mean(list(target_values.values()))),
            "targets": target_values,
        }
    expected = {
        (order, rung) for order in orders for rung in range(1, len(targets) + 1)
    }
    if set(table) != expected:
        raise ValueError(f"Expected {len(expected)} order/rung rows, found {len(table)}")
    for order, sequence in orders.items():
        if len(sequence) != len(targets) or set(sequence) != set(targets):
            raise ValueError(f"Order {order} is not a full target permutation")
    return orders, targets, table


def declutter(
    items: list[dict], *, gap: float, lower: float, upper: float
) -> list[dict]:
    items = sorted(items, key=lambda item: item["true"])
    for item in items:
        item["label"] = item["true"]
    for index in range(1, len(items)):
        items[index]["label"] = max(
            items[index]["label"], items[index - 1]["label"] + gap
        )
    if items[-1]["label"] > upper:
        items[-1]["label"] = upper
        for index in range(len(items) - 2, -1, -1):
            items[index]["label"] = min(
                items[index]["label"], items[index + 1]["label"] - gap
            )
    if items[0]["label"] < lower:
        shift = lower - items[0]["label"]
        for item in items:
            item["label"] += shift
    return items


def y_limits(
    table: dict[tuple[str, int], dict], targets: list[str], order: str
) -> tuple[float, float]:
    values = [
        row["targets"][target]
        for (row_order, _), row in table.items()
        if row_order == order
        for target in targets
    ]
    lower, upper = min(values), max(values)
    span = upper - lower
    pad = max(span * 0.08, upper * 0.01, 1e-5)
    return max(0.0, lower - pad), upper + pad


def plot_order(
    order: str,
    sequence: list[str],
    targets: list[str],
    table: dict[tuple[str, int], dict],
    limits: tuple[float, float],
    metric: str,
) -> None:
    rungs = np.arange(1, len(targets) + 1)
    labels = []
    entry_changes = []
    fig, ax = plt.subplots(figsize=(11.1, 6.0), dpi=200)

    for target in targets:
        values = np.array(
            [table[order, int(rung)]["targets"][target] for rung in rungs]
        )
        entry = sequence.index(target) + 1
        for left in range(1, len(targets)):
            in_training_at_right = left + 1 >= entry
            ax.plot(
                [left, left + 1],
                values[left - 1 : left + 1],
                color=BLUE if in_training_at_right else GRAY,
                lw=1.9 if in_training_at_right else 1.45,
                ls="-" if in_training_at_right else (0, (4, 2)),
                alpha=0.82 if in_training_at_right else 0.65,
                zorder=3,
                solid_capstyle="round",
            )
        for rung, value in zip(rungs, values):
            in_training = rung >= entry
            size = 92 if rung == entry and entry > 1 else 26
            ax.scatter(
                [rung],
                [value],
                s=size,
                facecolor=BLUE if in_training else "white",
                edgecolor="white" if rung == entry and entry > 1 else (
                    BLUE if in_training else GRAY
                ),
                linewidth=1.5 if rung == entry and entry > 1 else 1.25,
                zorder=6,
            )
        if entry > 1:
            entry_changes.append((target, values[entry - 1] - values[entry - 2]))
        labels.append({"true": values[-1], "text": SHORT[target]})

    means = np.array([table[order, int(rung)]["target_mean"] for rung in rungs])
    ax.plot(
        rungs,
        means,
        color=INK,
        lw=2.8,
        marker="s",
        ms=6,
        markerfacecolor=INK,
        markeredgecolor="white",
        markeredgewidth=1.1,
        zorder=8,
    )
    ax.annotate(
        "mean (all 9)",
        xy=(rungs[0] - 0.08, means[0]),
        ha="right",
        va="center",
        fontsize=9,
        color=INK,
        fontweight="bold",
    )

    span = limits[1] - limits[0]
    label_gap = span * 0.045
    for item in declutter(
        labels, gap=label_gap, lower=limits[0] + label_gap / 2, upper=limits[1] - label_gap / 2
    ):
        ax.plot(
            [rungs[-1] + 0.07, rungs[-1] + 0.27],
            [item["true"], item["label"]],
            color=MUTED,
            lw=0.65,
            zorder=2,
        )
        ax.annotate(
            item["text"],
            xy=(rungs[-1] + 0.33, item["label"]),
            ha="left",
            va="center",
            fontsize=9.1,
            color=BLUE,
            fontweight="bold",
        )

    tick_labels = [SHORT[target] for target in sequence]
    tick_labels[-1] += "\n(all 9)"
    ax.set_xlim(0.45, len(targets) + 1.75)
    ax.set_ylim(*limits)
    ax.set_xticks(rungs)
    ax.set_xticklabels(tick_labels, rotation=28, ha="right", fontsize=8.6)
    ax.set_xlabel(
        "source graph added at this rung (cumulative GraphCL training mixture)",
        fontsize=10.4,
        color=INK,
    )
    ax.set_ylabel(METRICS[metric]["ylabel"], fontsize=10.4, color=INK)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title(
        f"SAMGPT native-objective ladder · order {order}",
        fontsize=13,
        color=INK,
        fontweight="bold",
        loc="left",
        pad=27,
    )
    ax.text(
        0,
        1.02,
        "each line = one fixed target graph  ·  unseen corruption/edge-drop view  ·  "
        "checkpoint frozen  ·  blue begins when target enters training",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        color=MUTED,
    )
    ax.legend(
        handles=[
            Line2D(
                [0], [0], color=BLUE, lw=1.9, marker="o", markerfacecolor=BLUE,
                markeredgecolor="white", ms=7, label="target in training mixture"
            ),
            Line2D(
                [0], [0], color=GRAY, lw=1.45, ls=(0, (4, 2)), marker="o",
                markerfacecolor="white", markeredgecolor=GRAY, ms=7,
                label="target still held out"
            ),
            Line2D(
                [0], [0], color=INK, lw=2.8, marker="s", markerfacecolor=INK,
                markeredgecolor="white", ms=7, label="mean over all 9 targets"
            ),
        ],
        loc="lower right",
        frameon=False,
        fontsize=8.7,
        handlelength=2.5,
    )
    fig.tight_layout()
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    stem = METRICS[metric]["stem"].format(order=order)
    for suffix in ("pdf", "png"):
        path = FIG_ROOT / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=220)
        print("wrote", path)
    plt.close(fig)

    print(f"\norder {order} entry {metric} changes (entry minus previous rung):")
    for target, change in entry_changes:
        print(f"  {SHORT[target]:14s} {change:+.6f}")
    print("  mean(all 9):", ", ".join(f"{value:.6f}" for value in means))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=tuple(METRICS), default="probability_margin")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    orders, targets, table = load(args.metric)
    for order in ("A", "B", "C"):
        plot_order(
            order,
            orders[order],
            targets,
            table,
            y_limits(table, targets, order),
            args.metric,
        )


if __name__ == "__main__":
    main()
