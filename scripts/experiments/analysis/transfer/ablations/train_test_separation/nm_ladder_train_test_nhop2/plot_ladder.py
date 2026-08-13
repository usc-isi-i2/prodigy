#!/usr/bin/env python3
"""Plot the split-aware, fair-two-hop neighbor-matching ladder."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "nm_ladder_train_test_nhop2.csv"
FIGURES = HERE / "figures"
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
XTICKS = [
    "Ukr-Rus", "+COVID-19", "+Midterm", "+COVID-pol.",
    "+Election '20", "+Ukr-Rus susp.", "+TwiBot-20", "+CP-HK\n(all eight)",
]
BLUE, GRAY, INK = "#2878c8", "#8c8b86", "#111111"
MUTED, GRID, CORAL = "#6f6e69", "#deddd7", "#d45d3a"


def load() -> dict[str, list[float]]:
    with DATA.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["rung"]) for row in rows] != list(range(1, 9)):
        raise ValueError(f"{DATA} must contain exactly rungs 1..8")
    return {graph: [float(row[graph]) for row in rows]
            for graph, _label, _entry in GRAPHS}


def spread(items: list[dict[str, object]], gap: float = 0.018):
    items.sort(key=lambda item: float(item["actual"]))
    for item in items:
        item["label_y"] = float(item["actual"])
    for index in range(1, len(items)):
        items[index]["label_y"] = max(
            float(items[index]["label_y"]),
            float(items[index - 1]["label_y"]) + gap,
        )
    if float(items[-1]["label_y"]) > 0.976:
        items[-1]["label_y"] = 0.976
        for index in range(len(items) - 2, -1, -1):
            items[index]["label_y"] = min(
                float(items[index]["label_y"]),
                float(items[index + 1]["label_y"]) - gap,
            )
    return items


def main() -> None:
    series = load()
    x = list(range(1, 9))
    mean = [sum(series[graph][index] for graph, _label, _entry in GRAPHS) / 8
            for index in range(8)]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(11.0, 5.9), dpi=200)
    labels = []
    for graph, label, entry in GRAPHS:
        y = series[graph]
        for left in range(7):
            present = left + 2 >= entry
            ax.plot(
                x[left:left + 2], y[left:left + 2],
                color=BLUE if present else GRAY,
                lw=1.9 if present else 1.35,
                ls="-" if present else (0, (4, 2)),
                alpha=0.95 if present else 0.78, zorder=3,
            )
        for index, xpos in enumerate(x):
            present, is_entry = xpos >= entry, xpos == entry and entry > 1
            ax.scatter(
                xpos, y[index], s=74 if is_entry else 23,
                facecolor=CORAL if is_entry else (BLUE if present else "white"),
                edgecolor="white" if is_entry else (BLUE if present else GRAY),
                linewidth=1.4, zorder=6,
            )
        if entry > 1:
            delta = y[entry - 1] - y[entry - 2]
            ax.annotate(
                f"+{delta:.3f}", xy=(entry, y[entry - 1]), xytext=(0, 9),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=8.2, color=CORAL, weight="bold", zorder=8,
            )
        labels.append({"actual": y[-1], "label": label, "value": y[-1]})

    ax.plot(
        x, mean, color=INK, lw=2.8, marker="s", ms=5.8,
        markerfacecolor=INK, markeredgecolor="white", markeredgewidth=1.0,
        zorder=7,
    )
    for index in (0, 7):
        ax.annotate(
            f"{mean[index]:.3f}", xy=(x[index], mean[index]),
            xytext=(0, -15), textcoords="offset points", ha="center",
            fontsize=8.5, color=INK, weight="bold",
        )
    for item in spread(labels):
        actual, label_y = float(item["actual"]), float(item["label_y"])
        ax.plot([8.05, 8.28], [actual, label_y], color=MUTED, lw=0.65, zorder=2)
        ax.text(
            8.34, label_y, f"{item['label']}  {float(item['value']):.3f}",
            ha="left", va="center", fontsize=8.8, color=BLUE, weight="bold",
        )

    ax.set(xlim=(0.55, 9.35), ylim=(0.675, 0.985), xticks=x,
           ylabel="held-out-edge NM AUC",
           xlabel="pretraining mixture (one graph source added per rung)")
    ax.set_xticklabels(XTICKS, fontsize=8.6)
    ax.set_yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    ax.tick_params(colors=MUTED, labelsize=8.8)
    ax.grid(axis="y", color=GRID, lw=0.75, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#bebdb7")
    ax.set_title(
        "Adding graph sources improves neighbor matching on unseen edges",
        loc="left", fontsize=13.0, color=INK, weight="bold", pad=25,
    )
    ax.text(
        0.0, 1.02,
        "disjoint 15% positive-edge holdout  ·  3-shot / 30-way  ·  fair 2-hop sampler  ·  40k steps",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9.2, color=MUTED,
    )
    ax.legend(handles=[
        Line2D([0], [0], color=GRAY, lw=1.35, ls=(0, (4, 2)), marker="o",
               markerfacecolor="white", markeredgecolor=GRAY,
               label="source not yet in pretraining"),
        Line2D([0], [0], color=BLUE, lw=1.9, marker="o",
               markerfacecolor=BLUE, markeredgecolor="white",
               label="source included in pretraining"),
        Line2D([0], [0], color=CORAL, lw=0, marker="o",
               markerfacecolor=CORAL, markeredgecolor="white",
               label="source-entry point (label = AUC jump)"),
        Line2D([0], [0], color=INK, lw=2.8, marker="s",
               markerfacecolor=INK, markeredgecolor="white",
               label="mean over all eight test graphs"),
    ], loc="lower right", frameon=False, fontsize=8.7, handlelength=2.4)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    stem = FIGURES / "nm_ladder_train_test_nhop2"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    jumps = [series[graph][entry - 1] - series[graph][entry - 2]
             for graph, _label, entry in GRAPHS if entry > 1]
    print(f"wrote {stem}.png and .pdf")
    print(f"entry jumps: {sum(value > 0 for value in jumps)}/{len(jumps)} positive; "
          f"mean={sum(jumps) / len(jumps):+.4f}")
    print(f"all-graph mean: {mean[0]:.4f} -> {mean[-1]:.4f}")


if __name__ == "__main__":
    main()
