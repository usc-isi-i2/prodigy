#!/usr/bin/env python3
"""Plot the nine-source Order D NM ladder with Facebook inserted at rung 6."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "orderD_ladder_9x9.csv"
FIGURES = HERE / "figures"

BLUE = "#2a78d6"
GRAY = "#8f8d87"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

GRAPHS = [
    ("ukr_rus_twitter", "Ukr-Rus", 1),
    ("covid19_twitter", "COVID-19", 2),
    ("midterm", "Midterm", 3),
    ("covid_political", "COVID-pol.", 4),
    ("election2020", "Election '20", 5),
    ("facebook_page_reference", "Facebook", 6),
    ("ukr_rus_suspended", "Ukr-Rus susp.", 7),
    ("twibot20", "TwiBot-20", 8),
    ("cp_hk_twitter", "CP-HK", 9),
]
RUNGS = list(range(1, 10))
XTICKS = [
    "ukr", "+covid", "+midterm", "+cov-pol", "+elec '20",
    "+facebook", "+ukr-susp", "+twibot", "+cp-hk\n(all 9)",
]


def load():
    with DATA.open(newline="") as handle:
        rows = {int(row["rung"]): row for row in csv.DictReader(handle)}
    if sorted(rows) != RUNGS:
        raise ValueError(f"expected rungs {RUNGS}, got {sorted(rows)}")
    return {
        key: [float(rows[rung][key]) for rung in RUNGS]
        for key, _, _ in GRAPHS
    }


def declutter(items, gap, lo, hi):
    items = sorted(items, key=lambda item: item["y_true"])
    for item in items:
        item["y_label"] = item["y_true"]
    for index in range(1, len(items)):
        if items[index]["y_label"] - items[index - 1]["y_label"] < gap:
            items[index]["y_label"] = items[index - 1]["y_label"] + gap
    if items[-1]["y_label"] > hi:
        items[-1]["y_label"] = hi
        for index in range(len(items) - 2, -1, -1):
            if items[index + 1]["y_label"] - items[index]["y_label"] < gap:
                items[index]["y_label"] = items[index + 1]["y_label"] - gap
    return items


def main():
    series = load()
    x = list(range(len(RUNGS)))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(12.0, 6.3), dpi=200)

    labels = []
    for key, label, entry in GRAPHS:
        values = series[key]
        entry_index = entry - 1
        for index in range(len(x) - 1):
            in_training = index + 1 >= entry_index
            ax.plot(
                [x[index], x[index + 1]],
                [values[index], values[index + 1]],
                color=BLUE if in_training else GRAY,
                lw=1.9 if in_training else 1.5,
                ls="-" if in_training else (0, (4, 2)),
                zorder=3,
                solid_capstyle="round",
            )
        for index in range(len(x)):
            in_training = index >= entry_index
            if index == entry_index and entry > 1:
                ax.scatter(
                    [x[index]], [values[index]], s=95, facecolor=BLUE,
                    edgecolor="white", linewidth=1.6, zorder=6,
                )
            else:
                ax.scatter(
                    [x[index]], [values[index]], s=24,
                    facecolor=BLUE if in_training else "white",
                    edgecolor=BLUE if in_training else GRAY,
                    linewidth=1.4, zorder=5,
                )

        delta = values[entry_index] - values[entry_index - 1] if entry > 1 else 0.0
        show_delta = delta >= 0.03 or key == "facebook_page_reference"
        text = f"{label}   +{delta:.3f}" if show_delta else label
        labels.append({"y_true": values[-1], "text": text})

    mean = [sum(series[key][index] for key, _, _ in GRAPHS) / 9 for index in x]
    ax.plot(
        x, mean, color=INK, lw=2.8, zorder=8, marker="s", ms=6,
        markerfacecolor=INK, markeredgecolor="white", markeredgewidth=1.2,
    )
    ax.annotate(
        "mean (all 9)", xy=(x[0] - 0.06, mean[0]), ha="right", va="center",
        fontsize=9, color=INK, fontweight="bold",
    )

    facebook = series["facebook_page_reference"]
    before, after = facebook[4], facebook[5]
    error_reduction = (after - before) / (1 - before)
    ax.annotate(
        f"Facebook enters: {before:.3f} to {after:.3f} AUC\n"
        f"+{after - before:.3f} AUC  ·  {error_reduction:.0%} fewer misranked pairs",
        xy=(x[5], after), xytext=(5.35, 0.958),
        ha="left", va="top", fontsize=9.2, color=INK,
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.9},
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
              "edgecolor": GRID, "linewidth": 0.8},
        zorder=10,
    )

    right_x = x[-1]
    for item in declutter(labels, gap=0.016, lo=0.848, hi=0.998):
        ax.plot(
            [right_x + 0.06, right_x + 0.28],
            [item["y_true"], item["y_label"]],
            color=MUTED, lw=0.6, zorder=2,
        )
        ax.annotate(
            item["text"], xy=(right_x + 0.34, item["y_label"]),
            ha="left", va="center", fontsize=9.1, color=BLUE,
            fontweight="bold",
        )

    ax.set_xlim(-0.62, 10.0)
    ax.set_ylim(0.70, 1.01)
    ax.set_xticks(x)
    ax.set_xticklabels(XTICKS, fontsize=9.0)
    ax.set_xlabel(
        "SSL pre-training graph  (one source added per rung, merge grows to the right)",
        fontsize=10.5, color=INK,
    )
    ax.set_ylabel("NM AUC  (3-shot, 30-way)", fontsize=10.5, color=INK)
    ax.set_yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title(
        "Order D: Facebook enters at rung 6 and keeps its gain",
        fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=26,
    )
    ax.text(
        0.0, 1.02,
        "NM  3-shot / 30-way  ·  matched step 40k  ·  within-balanced sampling  ·  "
        "+d = out-of-dist. gain at entry rung",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=MUTED,
    )

    handles = [
        Line2D([0], [0], color=BLUE, lw=1.9, marker="o", markerfacecolor=BLUE,
               markeredgecolor="white", ms=7, label="in training (in-dist.)"),
        Line2D([0], [0], color=GRAY, lw=1.5, ls=(0, (4, 2)), marker="o",
               markerfacecolor="white", markeredgecolor=GRAY, ms=7,
               label="held out (out-of-dist.)"),
        Line2D([0], [0], color=INK, lw=2.8, marker="s", markerfacecolor=INK,
               markeredgecolor="white", ms=7, label="mean (all 9 graphs)"),
    ]
    ax.legend(
        handles=handles, loc="lower right", frameon=False, fontsize=9,
        handlelength=2.4, borderaxespad=0.6,
    )

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        output = FIGURES / f"nm_ladder_orderD_trajectory.{extension}"
        fig.savefig(output, bbox_inches="tight")
        print(f"wrote {output}")

    final = facebook[-1]
    print(f"Facebook entry gain: {before:.6f} -> {after:.6f} "
          f"(+{after - before:.6f}; {error_reduction:.1%} error reduction)")
    print(f"Facebook at rung 9: {final:.6f}; retained gain={final - before:+.6f}")
    print("mean(all 9): " + ", ".join(f"{value:.4f}" for value in mean))


if __name__ == "__main__":
    main()
