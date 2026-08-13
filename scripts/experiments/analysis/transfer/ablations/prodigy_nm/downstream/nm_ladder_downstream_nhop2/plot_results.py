#!/usr/bin/env python3
"""Plot the fair-two-hop ladder downstream results.

The three figures answer separate registered questions:

1. Do target metrics jump when their source graph first enters the ladder?
2. What do the complete rung-by-rung trajectories look like?
3. How do schedule/split/exposure variants differ from matched-40k training?

All summaries are descriptive paired measurements from one training seed.  Error
bars are deliberately avoided because the task-by-graph cells are not independent
replicates and the evaluation episode set is fixed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data"
DEFAULT_FIGURES = HERE / "figures"

INK = "#161616"
MUTED = "#77746e"
GRID = "#deddd7"
BLUE = "#2a78d6"
YELLOW = "#d99a00"
MAGENTA = "#d96c9a"
GREEN = "#16823b"
VIOLET = "#5a49ac"
CORAL = "#d15c45"
GRAY = "#8b8984"

TRAJECTORIES = [
    ("matched40k", "A", "Matched 40k · A"),
    ("sequential", "A", "Sequential · A"),
    ("split", "A", "Split-aware · A"),
    ("fixed10k", "A", "Fixed 10k/source · A"),
    ("fixed10k", "C", "Fixed 10k/source · C"),
]

VARIANT_LABELS = {
    "fixed10k": "Fixed 10k/source",
    "sequential": "Sequential",
    "split": "Split-aware",
}

TASK_LABELS = {
    "classification": "Node classification",
    "static_lp": "Static link prediction",
}

DATASET_LABELS = {
    "covid_political": "COVID political",
    "election2020": "Election 2020",
    "ukr_rus_suspended": "UKR/RUS suspended",
    "twibot20": "TwiBot-20",
    "ukr_rus_twitter": "UKR/RUS",
    "covid19_twitter": "COVID-19",
    "midterm": "US midterm",
    "cp_hk_twitter": "CP/HK",
}

DATASET_COLORS = {
    "covid_political": BLUE,
    "election2020": YELLOW,
    "ukr_rus_suspended": MAGENTA,
    "twibot20": GREEN,
    "ukr_rus_twitter": BLUE,
    "covid19_twitter": YELLOW,
    "midterm": MAGENTA,
    "cp_hk_twitter": VIOLET,
}

ADDED_LABELS = {
    "ukr_rus": "UKR/RUS",
    "covid": "COVID",
    "midterm": "Midterm",
    "covid_political": "COVID-pol.",
    "election2020": "Election",
    "ukr_rus_suspended": "UKR-susp.",
    "twibot20": "TwiBot-20",
    "cp_hk": "CP/HK",
}

ROLE_ORDER = ["heldout", "newcomer", "incumbent"]
ROLE_LABELS = {"heldout": "Held out", "newcomer": "Newcomer", "incumbent": "Incumbent"}
ROLE_COLORS = {"heldout": GRAY, "newcomer": CORAL, "incumbent": BLUE}
ROLE_MARKERS = {"heldout": "o", "newcomer": "^", "incumbent": "s"}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_results(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    entry = pd.read_csv(data_dir / "entry_jumps.csv")
    long = pd.read_csv(data_dir / "downstream_long.csv")
    paired = pd.read_csv(data_dir / "paired_to_matched40k.csv")

    if len(entry) != 40:
        raise ValueError(f"expected 40 entry jumps, found {len(entry)}")
    if len(paired) != 216:
        raise ValueError(f"expected 216 controlled pairs, found {len(paired)}")
    if long["logical_id"].nunique() != 40:
        raise ValueError(f"expected 40 logical ladder rows, found {long['logical_id'].nunique()}")
    if not set(entry["task"]) == set(TASK_LABELS):
        raise ValueError(f"unexpected entry tasks: {sorted(entry['task'].unique())}")
    return entry, long, paired


def entry_summary(entry: pd.DataFrame) -> pd.DataFrame:
    labels = {(variant, order): label for variant, order, label in TRAJECTORIES}
    rows: list[dict[str, object]] = []
    for (task, variant, order), group in entry.groupby(["task", "variant", "order"], sort=False):
        rows.append(
            {
                "task": task,
                "variant": variant,
                "order": order,
                "trajectory": labels[(variant, order)],
                "n": len(group),
                "positive": int((group["delta"] > 0).sum()),
                "mean": group["delta"].mean(),
                "minimum": group["delta"].min(),
                "maximum": group["delta"].max(),
            }
        )
    return pd.DataFrame(rows)


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_entry_jumps(entry: pd.DataFrame, output_dir: Path) -> None:
    summary = entry_summary(entry)
    order = [(v, o) for v, o, _ in TRAJECTORIES]
    labels = [label for _, _, label in TRAJECTORIES]

    x_min = min(-0.055, float(entry["delta"].min()) - 0.012)
    x_max = max(0.225, float(entry["delta"].max()) + 0.012)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.5), sharex=True, sharey=False)

    for ax, task in zip(axes, TASK_LABELS, strict=True):
        task_rows = entry[entry["task"] == task]
        overall_n = len(task_rows)
        overall_positive = int((task_rows["delta"] > 0).sum())
        overall_mean = task_rows["delta"].mean()

        ax.axvline(0, color=INK, linewidth=1.1, zorder=0)
        ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
        row_labels = []
        for y, ((variant, order_name), trajectory_label) in enumerate(zip(order, labels, strict=True)):
            group = task_rows[(task_rows["variant"] == variant) & (task_rows["order"] == order_name)]
            stats = summary[
                (summary["task"] == task)
                & (summary["variant"] == variant)
                & (summary["order"] == order_name)
            ].iloc[0]
            ax.hlines(y, stats["minimum"], stats["maximum"], color=GRAY, linewidth=1.4, zorder=1)
            offsets = np.linspace(-0.105, 0.105, len(group)) if len(group) > 1 else np.array([0.0])
            ax.scatter(
                group["delta"],
                y + offsets,
                s=32,
                facecolor="white",
                edgecolor=GRAY,
                linewidth=1.3,
                zorder=3,
            )
            ax.scatter(stats["mean"], y, marker="D", s=58, color=BLUE, edgecolor="white", linewidth=0.7, zorder=4)
            row_labels.append(
                f"{trajectory_label}\n{int(stats['positive'])}/{int(stats['n'])} positive · mean {stats['mean']:+.3f}"
            )

        ax.set_title(
            f"{TASK_LABELS[task]}\n{overall_positive}/{overall_n} positive · overall mean {overall_mean:+.3f}",
            loc="left",
            fontweight="bold",
        )
        ax.set_xlim(x_min, x_max)
        ax.set_yticks(range(len(row_labels)), row_labels)
        ax.invert_yaxis()
        ax.set_xlabel("Entry jump in ROC-AUC (after − before)")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Static LP improves at graph entry; classification is mixed",
        x=0.06,
        ha="left",
        y=1.01,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.955,
        "Dots are dataset entry events; diamonds are trajectory means. Shared x-scale; one training seed.",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.04, 0.02, 1, 0.91), w_pad=2.0)
    _save(fig, output_dir / "entry_jumps")


def _plot_trajectory_series(ax: plt.Axes, group: pd.DataFrame, color: str) -> None:
    group = group.sort_values("rung")
    x = group["rung"].to_numpy(dtype=float)
    y = group["value"].to_numpy(dtype=float)
    entry = int(group["entry_rung"].iloc[0]) if pd.notna(group["entry_rung"].iloc[0]) else 99

    pre = x < entry
    if pre.any():
        ax.plot(x[pre], y[pre], color=color, linewidth=1.25, linestyle=(0, (3, 2)), alpha=0.72)
    post_start = max(0, int(np.searchsorted(x, entry)) - 1)
    if entry <= x.max():
        ax.plot(x[post_start:], y[post_start:], color=color, linewidth=1.85, alpha=0.95)
        entry_row = group[group["rung"] == entry]
        if not entry_row.empty:
            ax.scatter(entry, entry_row["value"].iloc[0], s=28, marker="o", color=color, edgecolor="white", linewidth=0.6, zorder=4)
    else:
        ax.plot(x, y, color=color, linewidth=1.25, linestyle=(0, (3, 2)), alpha=0.72)


def plot_trajectories(long: pd.DataFrame, entry: pd.DataFrame, output_dir: Path) -> None:
    primary = long[long["primary"].astype(bool)].copy()
    summary = entry_summary(entry).set_index(["task", "variant", "order"])
    fig, axes = plt.subplots(2, 5, figsize=(18.8, 8.6), sharex=False, sharey="row")

    for row, task in enumerate(TASK_LABELS):
        task_data = primary[primary["task"] == task]
        datasets = list(task_data["dataset"].drop_duplicates())
        for col, (variant, order_name, label) in enumerate(TRAJECTORIES):
            ax = axes[row, col]
            panel = task_data[(task_data["variant"] == variant) & (task_data["order"] == order_name)]
            added = panel[["rung", "added"]].drop_duplicates().sort_values("rung")
            for dataset in datasets:
                group = panel[panel["dataset"] == dataset]
                _plot_trajectory_series(ax, group, DATASET_COLORS[dataset])

            stats = summary.loc[(task, variant, order_name)]
            ax.set_title(
                f"{label}\nentry mean {stats['mean']:+.3f} · {int(stats['positive'])}/{int(stats['n'])} positive",
                loc="left",
                fontweight="bold",
                fontsize=9.6,
            )
            ax.set_xticks(added["rung"], [ADDED_LABELS[a] for a in added["added"]], rotation=36, ha="right")
            ax.set_xlim(0.7, 8.3)
            ax.grid(axis="y", color=GRID, linewidth=0.65)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel(f"{TASK_LABELS[task]} ROC-AUC")
            if row == 1:
                ax.set_xlabel("Source added at rung")

        handles = [
            Line2D([0], [0], color=DATASET_COLORS[d], linewidth=2.0, marker="o", markersize=4, label=DATASET_LABELS[d])
            for d in datasets
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.905 if row == 0 else 0.465),
            ncol=len(handles),
            frameon=False,
            fontsize=8.5,
            handlelength=2.2,
            columnspacing=1.5,
        )

    fig.suptitle(
        "Downstream performance across all eight ladder rungs",
        x=0.055,
        ha="left",
        y=0.995,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.952,
        "Dashed = before the target source enters; solid = after entry; circles mark entry. The shared fixed-A8/C8 encoder appears in both logical trajectories.",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.035, 0.02, 1, 0.89), h_pad=5.2, w_pad=1.25)
    _save(fig, output_dir / "rung_trajectories")


def _jitter(group: pd.DataFrame) -> np.ndarray:
    # Stable, symmetric placement without implying random sampling.
    keys = group["rung"].astype(str) + "|" + group["dataset"].astype(str)
    ranks = keys.rank(method="first").to_numpy() - 1
    if len(group) == 1:
        return np.zeros(1)
    return (ranks / (len(group) - 1) - 0.5) * 0.34


def plot_controlled_deltas(paired: pd.DataFrame, output_dir: Path) -> None:
    variants = ["fixed10k", "split", "sequential"]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.6), sharey="row", sharex=True)

    for row, task in enumerate(TASK_LABELS):
        for col, variant in enumerate(variants):
            ax = axes[row, col]
            panel = paired[(paired["task"] == task) & (paired["variant"] == variant)]
            ax.axhline(0, color=INK, linewidth=1.0, zorder=0)
            ax.grid(axis="y", color=GRID, linewidth=0.65, zorder=0)
            for x_pos, role in enumerate(ROLE_ORDER):
                group = panel[panel["role"] == role]
                values = group["delta_vs_matched40k"].to_numpy()
                ax.scatter(
                    x_pos + _jitter(group),
                    values,
                    s=23,
                    marker=ROLE_MARKERS[role],
                    facecolor="white",
                    edgecolor=ROLE_COLORS[role],
                    linewidth=0.9,
                    alpha=0.72,
                    zorder=2,
                )
                mean = float(values.mean())
                ax.plot([x_pos - 0.24, x_pos + 0.24], [mean, mean], color=ROLE_COLORS[role], linewidth=3.0, zorder=3)
                ax.text(
                    x_pos,
                    mean + (0.010 if task == "static_lp" else 0.005),
                    f"{mean:+.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.8,
                    color=INK,
                )

            ax.set_title(VARIANT_LABELS[variant], loc="left", fontweight="bold")
            ax.set_xticks(range(len(ROLE_ORDER)), [ROLE_LABELS[r] for r in ROLE_ORDER])
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel(f"{TASK_LABELS[task]}\nΔ ROC-AUC vs matched 40k")
            if row == 0:
                ax.set_ylim(-0.115, 0.065)
            else:
                ax.set_ylim(-0.275, 0.065)

    fig.suptitle(
        "Split-aware and fixed-exposure variants track matched 40k; sequential static LP does not",
        x=0.065,
        ha="left",
        y=0.995,
        fontsize=14.5,
        fontweight="bold",
    )
    fig.text(
        0.065,
        0.948,
        "Every dot is a paired graph × rung cell. Horizontal bars are descriptive means, not confidence intervals.",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.045, 0.035, 1, 0.91), h_pad=2.6, w_pad=1.5)
    _save(fig, output_dir / "controlled_vs_matched40k")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _style()
    entry, long, paired = load_results(args.data_dir)
    plot_entry_jumps(entry, args.output_dir)
    plot_trajectories(long, entry, args.output_dir)
    plot_controlled_deltas(paired, args.output_dir)
    print(f"wrote 3 PNG/PDF figure pairs to {args.output_dir}")


if __name__ == "__main__":
    main()
