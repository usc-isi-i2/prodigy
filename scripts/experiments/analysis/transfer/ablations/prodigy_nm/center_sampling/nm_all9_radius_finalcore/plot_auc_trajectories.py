#!/usr/bin/env python3
"""Plot seed-0 10k validation ROC-AUC trajectories for radius sampling arms."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT = HERE / "data" / "validation_auc_seed0_10k.csv"
OUT = HERE / "figures" / "validation_auc_seed0_10k.png"
STEPS = [2500, 5000, 7500, 10000]
ARMS = ["global", "radius_mix", "close_only", "distance_stratified"]
LABELS = {
    "global": "Global",
    "radius_mix": "Radius mix",
    "close_only": "Close only",
    "distance_stratified": "Stratified",
}
COLORS = {
    "global": "#0072B2",
    "radius_mix": "#D55E00",
    "close_only": "#009E73",
    "distance_stratified": "#CC79A7",
}
MARKERS = {
    "global": "o",
    "radius_mix": "s",
    "close_only": "^",
    "distance_stratified": "D",
}
PANEL_IDS = ["primary_macro", "radius2", "radius3", "global", "within_source"]
PANEL_LABELS = {
    "primary_macro": "Primary-panel macro",
    "radius2": "Radius 2",
    "radius3": "Radius 3",
    "global": "Global",
    "within_source": "Within-source",
}


def load_panels():
    data = pd.read_csv(INPUT)
    expected = {
        (arm, 0, step, panel)
        for arm in ARMS
        for step in STEPS
        for panel in ("radius2", "radius3", "global", "within_source")
    }
    observed = set(
        data[["arm", "seed", "checkpoint_step", "panel"]].itertuples(
            index=False, name=None
        )
    )
    if observed != expected or len(data) != len(expected):
        raise ValueError("input must contain each arm x step x panel cell exactly once")
    macro = (
        data[data["panel"].isin(("radius2", "radius3", "global"))]
        .groupby(["arm", "seed", "checkpoint_step"], as_index=False)["roc_auc"]
        .mean()
    )
    macro["panel"] = "primary_macro"
    data = pd.concat([data, macro], ignore_index=True)
    panels = {}
    for panel in PANEL_IDS:
        panels[PANEL_LABELS[panel]] = {}
        for arm in ARMS:
            panels[PANEL_LABELS[panel]][arm] = (
                data[(data["panel"] == panel) & (data["arm"] == arm)]
                .set_index("checkpoint_step")["roc_auc"]
                .reindex(STEPS)
                .tolist()
            )
    return panels


PANELS = load_panels()


def spread_labels(values, low, high):
    gap = (high - low) * 0.075
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    placed = []
    for index, target in ordered:
        value = max(target, placed[-1][1] + gap) if placed else target
        placed.append([index, value])
    overflow = placed[-1][1] - high
    if overflow > 0:
        for item in placed:
            item[1] -= overflow
    for pos in range(len(placed) - 2, -1, -1):
        placed[pos][1] = min(placed[pos][1], placed[pos + 1][1] - gap)
    underflow = low - placed[0][1]
    if underflow > 0:
        for item in placed:
            item[1] += underflow
    return dict(placed)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.0))
axes = axes.ravel()

for ax, (panel, series) in zip(axes, PANELS.items()):
    all_values = [value for arm in ARMS for value in series[arm]]
    span = max(all_values) - min(all_values)
    pad = max(0.0018, span * 0.18)
    ymin = max(0.5, min(all_values) - pad)
    ymax = min(1.0, max(all_values) + pad)

    for arm in ARMS:
        ax.plot(
            STEPS,
            series[arm],
            color=COLORS[arm],
            marker=MARKERS[arm],
            linewidth=2.2,
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=3,
        )

    ax.set_title(panel, pad=8)
    ax.set_xlim(2200, 11400)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(STEPS, ["2.5k", "5k", "7.5k", "10k"])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.grid(axis="y", color="#C7C7C7", linewidth=0.7, alpha=0.55)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#777777")
    ax.tick_params(color="#777777")
    ax.set_xlabel("Optimizer updates")

    final_values = [series[arm][-1] for arm in ARMS]
    positions = spread_labels(final_values, ymin + pad * 0.15, ymax - pad * 0.15)
    for index, arm in enumerate(ARMS):
        label_y = positions[index]
        ax.plot(
            [10050, 10350],
            [final_values[index], label_y],
            color=COLORS[arm],
            linewidth=0.7,
            alpha=0.7,
            clip_on=False,
        )
        ax.text(
            10420,
            label_y,
            f"{final_values[index]:.3f}",
            color=COLORS[arm],
            fontsize=8.5,
            va="center",
            ha="left",
            clip_on=False,
        )

axes[0].set_ylabel("Validation ROC-AUC")
axes[3].set_ylabel("Validation ROC-AUC")

legend_ax = axes[-1]
legend_ax.axis("off")
handles = [
    Line2D(
        [0],
        [0],
        color=COLORS[arm],
        marker=MARKERS[arm],
        linewidth=2.2,
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=0.6,
        label=LABELS[arm],
    )
    for arm in ARMS
]
legend_ax.legend(
    handles=handles,
    loc="center",
    frameon=False,
    fontsize=11,
    handlelength=2.8,
    labelspacing=1.1,
)

fig.suptitle(
    "All-nine radius experiment: validation ROC-AUC trajectories",
    fontsize=16,
    fontweight="normal",
    y=0.985,
)
fig.text(
    0.5,
    0.012,
    "Macro one-vs-rest ROC-AUC, seed 0. Each panel uses its own y-axis range. "
    "Primary macro averages Radius 2, Radius 3, and Global.",
    ha="center",
    va="bottom",
    fontsize=9,
    color="#555555",
)
fig.subplots_adjust(left=0.07, right=0.985, top=0.92, bottom=0.09, hspace=0.34, wspace=0.23)
fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight")
plt.close(fig)
print(OUT)
