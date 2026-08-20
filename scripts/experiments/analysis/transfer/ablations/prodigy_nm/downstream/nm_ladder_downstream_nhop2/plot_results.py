#!/usr/bin/env python3
"""Plot the fair-two-hop ladder downstream results.

The five figures answer separate registered questions:

1. Do target metrics jump when their source graph first enters the ladder?
2. What do the complete rung-by-rung trajectories look like?
3. How do schedule/split/exposure variants differ from matched-40k training?
4. What is the classification-only mean change when a graph enters the mixture?
5. How does mean classification F1 change along the fixed-exposure ladder?

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
NM_DATA = {
    ("matched40k", "A"): HERE.parent.parent / "context_depth/nm_ladder_nhop2/data/nm_ladder_nhop2_order_A_long.csv",
    ("sequential", "A"): HERE.parent.parent / "source_schedule/nm_ladder_sequential_nhop2/data/nm_ladder_sequential_nhop2_long.csv",
    ("split", "A"): HERE.parent.parent / "split_integrity/nm_ladder_train_test_nhop2/data/nm_ladder_train_test_nhop2_long.csv",
    ("fixed10k", "A"): HERE.parent.parent / "source_exposure/nm_ladder_fixed_exposure_nhop2/data/logical_results.csv",
    ("fixed10k", "C"): HERE.parent.parent / "source_exposure/nm_ladder_fixed_exposure_nhop2/data/logical_results.csv",
}
FINAL_CORE_ROOT = HERE.parents[3] / "matrices/cross_model/final_core/data"
FINAL_CORE_NM = FINAL_CORE_ROOT / "prodigy_final_core/log_recovered_metrics/physical_metrics.tsv"
FINAL_CORE_CLS = FINAL_CORE_ROOT / "classification_ladder/classification_long.tsv"
FINAL_CORE_ORDERS = ("A", "B", "C")
FINAL_CORE_RUNG1 = {"A": "ss_ukr_rus", "B": "ss_ukr_rus_suspended", "C": "ss_twibot20"}
HISTORICAL_1H_NM = (
    HERE.parents[3]
    / "ladders/prodigy_nm/robustness/nm_ladder_order_robustness/data/nm_ladder_order_robustness_long.csv"
)
HISTORICAL_1H_DOWNSTREAM = HERE.parent / "nm_ladder_downstream/data/nm_ladder_downstream_long.csv"

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
    "facebook_page_reference": "Facebook",
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


def plot_classification_mean_change(entry: pd.DataFrame, output_dir: Path) -> None:
    """Show classification entry changes without the static-LP panel."""
    cls = entry[entry["task"] == "classification"]
    summary = entry_summary(cls).set_index(["variant", "order"])
    trajectories = [(variant, order, label) for variant, order, label in TRAJECTORIES]

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    ax.axvline(0, color=INK, linewidth=1.15, zorder=0)
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)

    row_labels: list[str] = []
    for y, (variant, order_name, label) in enumerate(trajectories):
        group = cls[(cls["variant"] == variant) & (cls["order"] == order_name)].sort_values("dataset")
        stats = summary.loc[(variant, order_name)]
        offsets = np.linspace(-0.11, 0.11, len(group)) if len(group) > 1 else np.array([0.0])
        ax.scatter(
            group["delta"],
            y + offsets,
            s=42,
            facecolor="white",
            edgecolor=GRAY,
            linewidth=1.25,
            zorder=2,
        )
        ax.scatter(
            stats["mean"],
            y,
            marker="D",
            s=82,
            color=BLUE if variant == "matched40k" else CORAL if variant == "fixed10k" else GRAY,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.text(
            stats["mean"] + 0.003,
            y,
            f"{stats['mean']:+.3f}",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
        )
        compute = "fixed total compute" if variant == "matched40k" else (
            "fixed exposure/source" if variant == "fixed10k" else "control"
        )
        row_labels.append(f"{label}\n{compute} · {int(stats['positive'])}/{int(stats['n'])} positive")

    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.045, 0.05)
    ax.set_xlabel("Mean change in classification ROC-AUC at graph entry (after − before)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.set_title(
        "Adding a graph to the pretraining mixture does not consistently improve classification",
        loc="left",
        fontsize=13.5,
        fontweight="bold",
        pad=34,
    )
    ax.text(
        0,
        1.035,
        "Diamonds are trajectory means; open circles are individual graph-entry changes. One training seed.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout()
    _save(fig, output_dir / "classification_mean_entry_change")


def plot_fixed_exposure_mean_f1_ladder(long: pd.DataFrame, output_dir: Path) -> None:
    """Plot one macro-over-datasets F1 line for fixed exposure, canonical Order A."""
    panel = long[
        (long["task"] == "classification")
        & (long["metric"] == "f1")
        & (long["variant"] == "fixed10k")
        & (long["order"] == "A")
    ].copy()
    if panel.empty:
        raise ValueError("no fixed-exposure Order A classification F1 rows")

    means = panel.groupby("rung", as_index=False)["value"].mean()
    added = panel[["rung", "added"]].drop_duplicates().sort_values("rung")
    if len(means) != 8 or panel["dataset"].nunique() != 4:
        raise ValueError("expected eight rungs and four classification datasets")

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(
        means["rung"],
        means["value"],
        color=BLUE,
        linewidth=2.7,
        marker="o",
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=0.9,
        zorder=3,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.75)
    ax.set_xticks(added["rung"], [ADDED_LABELS[a] for a in added["added"]], rotation=32, ha="right")
    ax.set_xlim(0.75, 8.25)
    padding = 0.008
    ax.set_ylim(float(means["value"].min()) - padding, float(means["value"].max()) + padding)
    ax.set_xlabel("Source added at rung")
    ax.set_ylabel("Mean classification F1")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.set_title(
        "Classification does not improve as the fixed-exposure mixture grows",
        loc="left",
        fontsize=14,
        fontweight="bold",
        pad=34,
    )
    ax.text(
        0,
        1.035,
        "Fixed 10k examples per active source · Order A · macro-average across four evaluation graphs",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9.2,
    )
    fig.tight_layout()
    _save(fig, output_dir / "fixed_exposure_classification_mean_f1_ladder")


def plot_classification_mean_ladders(long: pd.DataFrame, output_dir: Path) -> None:
    """Write one mean-over-datasets ladder for every trajectory and CLS metric."""
    metric_specs = [
        ("accuracy", "accuracy", "Accuracy", False),
        ("f1", "f1", "F1", False),
        ("roc_auc", "roc_auc", "ROC-AUC", False),
        ("accuracy_change", "accuracy", "ΔAccuracy from rung 1", True),
        ("f1_change", "f1", "ΔF1 from rung 1", True),
        ("roc_auc_change", "roc_auc", "ΔROC-AUC from rung 1", True),
    ]
    target_dir = output_dir / "classification_mean_ladders"

    for variant, order_name, trajectory_label in TRAJECTORIES:
        for output_metric, source_metric, metric_label, relative_to_rung1 in metric_specs:
            panel = long[
                (long["task"] == "classification")
                & (long["metric"] == source_metric)
                & (long["variant"] == variant)
                & (long["order"] == order_name)
            ].copy()
            means = panel.groupby("rung", as_index=False)["value"].mean()
            if relative_to_rung1:
                means["value"] -= float(means.loc[means["rung"] == 1, "value"].iloc[0])
            added = panel[["rung", "added"]].drop_duplicates().sort_values("rung")
            if len(means) != 8 or panel["dataset"].nunique() != 4:
                raise ValueError(
                    f"expected eight rungs and four datasets for {variant}/{order_name}/{output_metric}"
                )

            fig, ax = plt.subplots(figsize=(9.5, 5.5))
            ax.plot(
                means["rung"],
                means["value"],
                color=BLUE,
                linewidth=2.7,
                marker="o",
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.9,
                zorder=3,
            )
            ax.grid(axis="y", color=GRID, linewidth=0.75)
            if relative_to_rung1:
                ax.axhline(0, color=INK, linewidth=1.0, zorder=1)
            ax.set_xticks(
                added["rung"],
                [ADDED_LABELS[a] for a in added["added"]],
                rotation=32,
                ha="right",
            )
            ax.set_xlim(0.75, 8.25)
            span = float(means["value"].max() - means["value"].min())
            padding = max(0.008, span * 0.25)
            if relative_to_rung1:
                ax.set_ylim(float(means["value"].min()) - padding, float(means["value"].max()) + padding)
            else:
                ax.set_ylim(0.5, 1.0)
            ax.set_xlabel("Source added at rung")
            ax.set_ylabel(f"Mean classification {metric_label}")
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.set_title(
                f"Classification {metric_label} across mixture size",
                loc="left",
                fontsize=14,
                fontweight="bold",
                pad=34,
            )
            exposure_label = {
                "matched40k": "Fixed total compute (40k steps)",
                "sequential": "Sequential schedule · fixed total compute (40k steps)",
                "split": "Split-aware · fixed total compute (40k steps)",
                "fixed10k": "Fixed 10k examples per active source",
            }[variant]
            ax.text(
                0,
                1.035,
                f"{trajectory_label} · {exposure_label} · mean across four evaluation graphs",
                transform=ax.transAxes,
                color=MUTED,
                fontsize=9.2,
            )
            fig.tight_layout()
            stem = f"{variant}_order{order_name}_{output_metric}"
            _save(fig, target_dir / stem)


def plot_classification_auc_trajectories(long: pd.DataFrame, output_pdf: Path) -> None:
    """Plot all classification-dataset AUC trajectories in five side-by-side panels."""
    panel_data = long[(long["task"] == "classification") & (long["metric"] == "roc_auc")]
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.2), sharey=True)

    for ax, (variant, order_name, label) in zip(axes, TRAJECTORIES, strict=True):
        panel = panel_data[(panel_data["variant"] == variant) & (panel_data["order"] == order_name)]
        added = panel[["rung", "added"]].drop_duplicates().sort_values("rung")
        for dataset, group in panel.groupby("dataset", sort=False):
            group = group.sort_values("rung")
            ax.plot(
                group["rung"],
                group["value"],
                color=DATASET_COLORS[dataset],
                linewidth=2.0,
                marker="o",
                markersize=4.2,
                label=DATASET_LABELS[dataset],
            )
        ax.set_title(label, loc="left", fontweight="bold", fontsize=11)
        ax.set_xticks(
            added["rung"],
            [ADDED_LABELS[a] for a in added["added"]],
            rotation=38,
            ha="right",
        )
        ax.set_xlim(0.75, 8.25)
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Source added")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Classification ROC-AUC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Classification ROC-AUC across mixture size",
        x=0.04,
        y=0.995,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.945,
        "One line per evaluation graph · shared 0.5-1.0 y-axis",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.025, 0.02, 1, 0.86), w_pad=1.2)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def plot_classification_mean_auc_one_row(long: pd.DataFrame, output_pdf: Path) -> None:
    """Plot one mean classification AUC trajectory for each ladder variant."""
    panel_data = long[(long["task"] == "classification") & (long["metric"] == "roc_auc")]
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.2), sharey=True)

    for ax, (variant, order_name, label) in zip(axes, TRAJECTORIES, strict=True):
        panel = panel_data[(panel_data["variant"] == variant) & (panel_data["order"] == order_name)]
        means = panel.groupby("rung", as_index=False)["value"].mean()
        added = panel[["rung", "added"]].drop_duplicates().sort_values("rung")
        ax.plot(
            means["rung"],
            means["value"],
            color=BLUE,
            linewidth=2.5,
            marker="o",
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=0.7,
        )
        ax.set_title(label, loc="left", fontweight="bold", fontsize=11)
        ax.set_xticks(
            added["rung"],
            [ADDED_LABELS[a] for a in added["added"]],
            rotation=38,
            ha="right",
        )
        ax.set_xlim(0.75, 8.25)
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Source added")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean classification ROC-AUC")
    fig.suptitle(
        "Mean classification ROC-AUC across mixture size",
        x=0.04,
        y=0.995,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.945,
        "Mean across four evaluation graphs · shared 0.5-1.0 y-axis",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.025, 0.02, 1, 0.89), w_pad=1.2)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


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


def load_nm_trajectories() -> pd.DataFrame:
    """Normalize the five independently assembled NM ladders for the joint figure."""
    frames = []
    for (variant, order_name), path in NM_DATA.items():
        raw = pd.read_csv(path)
        if variant == "fixed10k":
            raw = raw[raw["order"] == order_name]
            frame = raw.rename(
                columns={"dataset": "dataset", "added_dataset": "added", "test_roc_auc": "value"}
            )[["rung", "added", "dataset", "entry_rung", "value"]]
            frame["added"] = frame["added"].replace(
                {"ukr_rus_twitter": "ukr_rus", "covid19_twitter": "covid", "cp_hk_twitter": "cp_hk"}
            )
        else:
            dataset_column = "test_graph"
            frame = raw.rename(columns={dataset_column: "dataset", "auc": "value"})[
                ["rung", "added", "dataset", "entry_rung", "value"]
            ]
        frame["variant"] = variant
        frame["order"] = order_name
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    expected = len(TRAJECTORIES) * 8 * 8
    if len(result) != expected:
        raise ValueError(f"expected {expected} NM trajectory cells, found {len(result)}")
    return result


def nm_entry_summary(nm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, order_name, dataset), group in nm.groupby(["variant", "order", "dataset"]):
        entry_rung = int(group["entry_rung"].iloc[0])
        before = group[group["rung"] == entry_rung - 1]
        after = group[group["rung"] == entry_rung]
        if before.empty:  # The first source has no pre-entry rung.
            continue
        rows.append(
            {"variant": variant, "order": order_name, "delta": float(after["value"].iloc[0] - before["value"].iloc[0])}
        )
    deltas = pd.DataFrame(rows)
    return deltas.groupby(["variant", "order"])["delta"].agg(
        mean="mean", n="size", positive=lambda values: int((values > 0).sum())
    )


def _final_core_model(order_name: str, rung: int) -> str:
    if rung == 1:
        return FINAL_CORE_RUNG1[order_name]
    if rung == 9:
        return "all9"
    return f"ord{order_name}_r{rung}"


def load_final_core_2500() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Load three-seed, nine-rung fixed-compute NM and classification trajectories."""
    nm_raw = pd.read_csv(FINAL_CORE_NM, sep="\t")
    cls_raw = pd.read_csv(FINAL_CORE_CLS, sep="\t")
    sources = cls_raw[["model_id", "sources"]].drop_duplicates()
    if sources.groupby("model_id")["sources"].nunique().max() != 1:
        raise ValueError("inconsistent final-core source lists")
    source_map = dict(zip(sources["model_id"], sources["sources"], strict=True))

    frames: dict[str, list[pd.DataFrame]] = {"nm": [], "classification": []}
    for order_name in FINAL_CORE_ORDERS:
        models = [_final_core_model(order_name, rung) for rung in range(1, 10)]
        for task, raw, seed_col, dataset_col, value_col in (
            ("nm", nm_raw, "seed", "target", "roc_auc_ovr_macro_logged"),
            ("classification", cls_raw, "training_seed", "dataset", "roc_auc"),
        ):
            panel = raw[raw["model_id"].isin(models)].copy()
            panel["rung"] = panel["model_id"].map({model: rung for rung, model in enumerate(models, 1)})
            panel = panel.rename(columns={seed_col: "seed", dataset_col: "dataset", value_col: "value"})
            panel["order"] = order_name
            frames[task].append(panel[["seed", "order", "rung", "model_id", "dataset", "value"]])
    nm = pd.concat(frames["nm"], ignore_index=True)
    classification = pd.concat(frames["classification"], ignore_index=True)
    for name, frame, targets in (("NM", nm, 9), ("classification", classification, 5)):
        expected = 3 * 3 * 9 * targets
        if len(frame) != expected:
            raise ValueError(f"expected {expected} final-core {name} cells, found {len(frame)}")
    return nm, classification, source_map


def _plot_seeded_trajectory(ax: plt.Axes, group: pd.DataFrame, color: str, entry: int) -> None:
    pivot = group.pivot(index="seed", columns="rung", values="value").sort_index(axis=1)
    x = pivot.columns.to_numpy(dtype=float)
    values = pivot.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    ax.fill_between(x, values.min(axis=0), values.max(axis=0), color=color, alpha=0.10, linewidth=0)
    pre = x < entry
    if pre.any():
        ax.plot(x[pre], mean[pre], color=color, linewidth=1.25, linestyle=(0, (3, 2)), alpha=0.72)
    post_start = max(0, int(np.searchsorted(x, entry)) - 1)
    ax.plot(x[post_start:], mean[post_start:], color=color, linewidth=1.85, alpha=0.95)
    entry_index = np.flatnonzero(x == entry)
    if len(entry_index):
        ax.scatter(entry, mean[entry_index[0]], s=28, marker="o", color=color,
                   edgecolor="white", linewidth=0.6, zorder=4)


def _final_core_entry_stats(panel: pd.DataFrame, source_map: dict[str, str]) -> tuple[float, int, int]:
    deltas = []
    for (order_name, dataset, seed), group in panel.groupby(["order", "dataset", "seed"]):
        entry = next(
            rung for rung in range(1, 10)
            if dataset in source_map[_final_core_model(order_name, rung)].split(",")
        )
        if entry == 1:
            continue
        by_rung = group.set_index("rung")["value"]
        deltas.append(float(by_rung.loc[entry] - by_rung.loc[entry - 1]))
    values = np.asarray(deltas)
    return float(values.mean()), int((values > 0).sum()), len(values)


def load_historical_1h_matched40k() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the historical one-hop matched-40k B/C NM and classification ladders."""
    nm = pd.read_csv(HISTORICAL_1H_NM).rename(columns={"test_graph": "dataset", "auc": "value"})
    nm = nm[nm["order"].isin(("B", "C"))][
        ["order", "rung", "added", "dataset", "entry_rung", "value"]
    ]
    downstream = pd.read_csv(HISTORICAL_1H_DOWNSTREAM)
    classification = downstream[
        (downstream["order"].isin(("B", "C")))
        & (downstream["task"] == "pl")
        & (downstream["metric"] == "roc_auc")
        & downstream["primary"].astype(bool)
    ][["order", "rung", "added", "dataset", "entry_rung", "value"]].copy()
    for name, frame, targets in (("NM", nm, 8), ("classification", classification, 4)):
        expected = 2 * 8 * targets
        if len(frame) != expected:
            raise ValueError(f"expected {expected} historical one-hop {name} cells, found {len(frame)}")
    return nm, classification


def plot_trajectories(
    long: pd.DataFrame,
    entry: pd.DataFrame,
    output_dir: Path,
    *,
    match_nm_to_classification: bool = False,
    order_columns: bool = False,
    output_stem: str = "rung_trajectories",
) -> None:
    primary = long[long["primary"].astype(bool)].copy()
    summary = entry_summary(entry).set_index(["task", "variant", "order"])
    nm = load_nm_trajectories()
    nm_summary = nm_entry_summary(nm)
    final_nm, final_cls, final_sources = load_final_core_2500()
    historical_nm, historical_cls = load_historical_1h_matched40k()
    ncols = 10 if order_columns else 8
    fig, axes = plt.subplots(3, ncols, figsize=(36.0 if order_columns else 29.2, 12.2), sharex=False, sharey="row")
    if order_columns:
        column_for = {
            ("legacy", "matched40k", "A"): 0,
            ("legacy", "sequential", "A"): 1,
            ("legacy", "split", "A"): 2,
            ("legacy", "fixed10k", "A"): 3,
            ("final", "matched2500", "A"): 4,
            ("historical", "matched40k", "B"): 5,
            ("final", "matched2500", "B"): 6,
            ("historical", "matched40k", "C"): 7,
            ("final", "matched2500", "C"): 8,
            ("legacy", "fixed10k", "C"): 9,
        }
    else:
        column_for = {
            **{("legacy", variant, order_name): col for col, (variant, order_name, _) in enumerate(TRAJECTORIES)},
            **{("final", "matched2500", order_name): col for col, order_name in enumerate(FINAL_CORE_ORDERS, 5)},
        }

    for row, task in enumerate(TASK_LABELS):
        task_data = primary[primary["task"] == task]
        datasets = list(task_data["dataset"].drop_duplicates())
        for variant, order_name, label in TRAJECTORIES:
            col = column_for[("legacy", variant, order_name)]
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
        handles = [
            Line2D([0], [0], color=DATASET_COLORS[d], linewidth=2.0, marker="o", markersize=4, label=DATASET_LABELS[d])
            for d in datasets
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.905 if row == 0 else 0.61),
            ncol=len(handles),
            frameon=False,
            fontsize=8.5,
            handlelength=2.2,
            columnspacing=1.5,
        )

    row = 2
    datasets = list(nm["dataset"].drop_duplicates())
    if match_nm_to_classification:
        datasets = list(primary[primary["task"] == "classification"]["dataset"].drop_duplicates())
    for variant, order_name, label in TRAJECTORIES:
        col = column_for[("legacy", variant, order_name)]
        ax = axes[row, col]
        panel = nm[(nm["variant"] == variant) & (nm["order"] == order_name)]
        added = panel[["rung", "added"]].drop_duplicates().sort_values("rung")
        for dataset in datasets:
            _plot_trajectory_series(ax, panel[panel["dataset"] == dataset], DATASET_COLORS[dataset])
        stats = nm_summary.loc[(variant, order_name)]
        ax.set_title(
            f"{label}\nentry mean {stats['mean']:+.3f} · {int(stats['positive'])}/{int(stats['n'])} positive",
            loc="left", fontweight="bold", fontsize=9.6,
        )
        ax.set_xticks(added["rung"], [ADDED_LABELS[a] for a in added["added"]], rotation=36, ha="right")
        ax.set_xlim(0.7, 8.3)
        ax.grid(axis="y", color=GRID, linewidth=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.set_xlabel("Source added at rung")
        if col == 0:
            ax.set_ylabel("Neighbor matching ROC-AUC")
    handles = [
        Line2D([0], [0], color=DATASET_COLORS[d], linewidth=2.0, marker="o", markersize=4, label=DATASET_LABELS[d])
        for d in datasets
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.325), ncol=len(handles),
               frameon=False, fontsize=8.2, handlelength=2.0, columnspacing=1.1)

    final_added = {}
    for order_name in FINAL_CORE_ORDERS:
        additions = []
        previous: set[str] = set()
        for rung in range(1, 10):
            current = set(final_sources[_final_core_model(order_name, rung)].split(","))
            added = current - previous
            if len(added) != 1:
                raise ValueError(f"expected one added source for final-core {order_name}{rung}")
            additions.append(added.pop())
            previous = current
        final_added[order_name] = additions
    for order_name in FINAL_CORE_ORDERS:
        offset = column_for[("final", "matched2500", order_name)]
        label = f"Matched 2.5k · {order_name}"
        for row, (task_name, panel) in enumerate((("classification", final_cls), ("nm", final_nm))):
            target_row = 0 if task_name == "classification" else 2
            ax = axes[target_row, offset]
            order_panel = panel[panel["order"] == order_name]
            if task_name == "nm" and match_nm_to_classification:
                order_panel = order_panel[order_panel["dataset"].isin(final_cls["dataset"].unique())]
            for dataset, group in order_panel.groupby("dataset", sort=False):
                entry = next(
                    rung for rung in range(1, 10)
                    if dataset in final_sources[_final_core_model(order_name, rung)].split(",")
                )
                _plot_seeded_trajectory(ax, group, DATASET_COLORS.get(dataset, GRAY), entry)
            mean, positive, count = _final_core_entry_stats(order_panel, final_sources)
            ax.set_title(
                f"{label}\nentry mean {mean:+.3f} · {positive}/{count} positive",
                loc="left", fontweight="bold", fontsize=9.6,
            )
            ax.set_xticks(
                range(1, 10),
                [ADDED_LABELS.get(source, source) for source in final_added[order_name]],
                rotation=36, ha="right",
            )
            ax.set_xlim(0.7, 9.3)
            ax.grid(axis="y", color=GRID, linewidth=0.65)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            if target_row == 2:
                ax.set_xlabel("Source added at rung")

        ax = axes[1, offset]
        ax.set_title(f"{label}\nstatic LP not evaluated", loc="left", fontweight="bold", fontsize=9.6)
        ax.text(0.5, 0.5, "No repaired static-LP results", transform=ax.transAxes,
                ha="center", va="center", color=MUTED, fontsize=9)
        ax.set_xticks([])
        ax.grid(False)
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)

    if order_columns:
        classification_targets = list(
            primary[primary["task"] == "classification"]["dataset"].drop_duplicates()
        )
        for order_name in ("B", "C"):
            col = column_for[("historical", "matched40k", order_name)]
            label = f"Matched 40k · {order_name} · 1-hop"
            for target_row, panel in ((0, historical_cls), (2, historical_nm)):
                ax = axes[target_row, col]
                order_panel = panel[(panel["order"] == order_name) & panel["dataset"].isin(classification_targets)]
                deltas = []
                for dataset, group in order_panel.groupby("dataset", sort=False):
                    _plot_trajectory_series(ax, group, DATASET_COLORS[dataset])
                    entry_rung = int(group["entry_rung"].iloc[0])
                    if entry_rung > 1:
                        values = group.set_index("rung")["value"]
                        deltas.append(float(values.loc[entry_rung] - values.loc[entry_rung - 1]))
                delta_values = np.asarray(deltas)
                ax.set_title(
                    f"{label}\nentry mean {delta_values.mean():+.3f} · {int((delta_values > 0).sum())}/{len(delta_values)} positive",
                    loc="left", fontweight="bold", fontsize=9.6,
                )
                added = order_panel[["rung", "added"]].drop_duplicates().sort_values("rung")
                ax.set_xticks(added["rung"], [ADDED_LABELS[a] for a in added["added"]], rotation=36, ha="right")
                ax.set_xlim(0.7, 8.3)
                ax.grid(axis="y", color=GRID, linewidth=0.65)
                ax.spines[["top", "right"]].set_visible(False)
                ax.set_axisbelow(True)
                if target_row == 2:
                    ax.set_xlabel("Source added at rung")
            ax = axes[1, col]
            ax.set_title(f"{label}\nlegacy static LP invalid", loc="left", fontweight="bold", fontsize=9.6)
            ax.text(0.5, 0.5, "Old episodic static-LP scores are void", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED, fontsize=9)
            ax.set_xticks([])
            ax.grid(False)
            ax.spines[["top", "right", "bottom", "left"]].set_visible(False)

    divider_columns = (4, 6) if order_columns else (4,)
    for row in range(3):
        for divider_col in divider_columns:
            axes[row, divider_col].spines["right"].set_visible(True)
            axes[row, divider_col].spines["right"].set_color(GRID)
            axes[row, divider_col].spines["right"].set_linewidth(1.4)

    fig.suptitle(
        "Downstream performance across matched-compute and fixed-exposure ladders",
        x=0.055,
        ha="left",
        y=0.995,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.952,
        (
            "Columns grouped by source order (A, then B, then C) · NM uses the same evaluation graphs as classification in each column."
            if order_columns
            else "Left: fair-two-hop eight-source ladders (one seed). Right: matched 2.5k nine-source ladders (three-seed means; bands show seed ranges)."
        ),
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.035, 0.02, 1, 0.89), h_pad=5.0, w_pad=1.25)
    _save(fig, output_dir / output_stem)


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
    plot_classification_mean_change(entry, args.output_dir)
    plot_fixed_exposure_mean_f1_ladder(long, args.output_dir)
    plot_classification_mean_ladders(long, args.output_dir)
    plot_trajectories(long, entry, args.output_dir)
    plot_trajectories(
        long,
        entry,
        args.output_dir,
        match_nm_to_classification=True,
        order_columns=True,
        output_stem="rung_trajectories_matched_graphs_by_order",
    )
    plot_controlled_deltas(paired, args.output_dir)
    print(f"wrote 6 core plus 15 classification-ladder PNG/PDF figure pairs to {args.output_dir}")


if __name__ == "__main__":
    main()
