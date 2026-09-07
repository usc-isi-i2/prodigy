#!/usr/bin/env python3
"""Plot matched A/B/C native-pretraining mixture ladders for downstream CLS."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[5]
FINAL_CORE_SETUP = REPO / "scripts/experiments/setup/final_core"
sys.path.insert(0, str(FINAL_CORE_SETUP))
from core_plan import ORDERS, build_models  # noqa: E402


TARGETS = (
    "covid_political",
    "election2020",
    "facebook_page_reference",
    "twibot20",
    "ukr_rus_suspended",
)
TARGET_LABELS = {
    "covid_political": "COVID political",
    "election2020": "Election2020",
    "facebook_page_reference": "Facebook pages",
    "twibot20": "TwiBot20",
    "ukr_rus_suspended": "UKR/RUS suspended",
}
SOURCE_LABELS = {
    "ukr_rus": "Ukraine/Russia",
    "covid": "COVID",
    "midterm": "Midterm",
    "covid_political": "COVID political",
    "election2020": "Election2020",
    "ukr_rus_suspended": "UKR/RUS susp.",
    "twibot20": "TwiBot20",
    "cp_hk": "Hong Kong",
    "facebook_page_reference": "Facebook",
}
MODEL_ROWS = (
    ("PRODIGY", "NM · 2,500 updates · 3 seeds"),
    ("VISION", "feature similarity · 2,500 updates · seed 0 · odd rungs"),
    ("SAMGPT", "GraphCL · 500 updates · 3 seeds"),
)


def model_for(order: str, rung: int) -> str:
    wanted = frozenset(ORDERS[order][:rung])
    matches = [model.model_id for model in build_models() if frozenset(model.sources) == wanted]
    if len(matches) != 1:
        raise ValueError(f"{order}{rung}: expected one final-core model, got {matches}")
    return matches[0]


def decorate(frame: pd.DataFrame, model: str, seeds: int, step: int, objective: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["model"] = model
    frame["training_seeds"] = seeds
    frame["checkpoint_step"] = step
    frame["objective"] = objective
    frame["added"] = [ORDERS[order][int(rung) - 1] for order, rung in zip(frame.order, frame.rung)]
    return frame[
        ["model", "order", "rung", "added", "target", "roc_auc", "training_seeds", "checkpoint_step", "objective"]
    ]


def load_prodigy() -> pd.DataFrame:
    path = (
        REPO
        / "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/"
        "data/classification_ladder/classification_long.tsv"
    )
    raw = pd.read_csv(path, sep="\t")
    raw = raw[raw.dataset.isin(TARGETS) & raw.checkpoint_step.eq(2500)].copy()
    rows = []
    for order in ORDERS:
        for rung in range(1, 10):
            model_id = model_for(order, rung)
            part = raw[raw.model_id.eq(model_id)]
            if part.training_seed.nunique() != 3 or set(part.dataset) != set(TARGETS):
                raise ValueError(f"PRODIGY {order}{rung}: incomplete common-target grid")
            for target, target_rows in part.groupby("dataset"):
                rows.append(
                    {"order": order, "rung": rung, "target": target, "roc_auc": target_rows.roc_auc.mean()}
                )
    return decorate(pd.DataFrame(rows), "PRODIGY", 3, 2500, "neighbor matching")


def load_vision() -> pd.DataFrame:
    path = ROOT / "data/vision_native_mixture_per_target.csv"
    raw = pd.read_csv(path)
    raw = raw[raw.checkpoint_step.eq(2500) & raw.dataset.isin(TARGETS)].copy()
    raw = raw.rename(columns={"dataset": "target"})
    expected = {(order, rung, target) for order in ORDERS for rung in (1, 3, 5, 7, 9) for target in TARGETS}
    observed = set(zip(raw.order, raw.rung, raw.target))
    if observed != expected:
        raise ValueError(f"VISION grid mismatch: missing={sorted(expected-observed)} extra={sorted(observed-expected)}")
    return decorate(raw[["order", "rung", "target", "roc_auc"]], "VISION", 1, 2500, "feature similarity")


def load_samgpt() -> pd.DataFrame:
    path = (
        REPO
        / "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/"
        "data/samgpt_downstream_cls/three_seed_mean.csv"
    )
    raw = pd.read_csv(path)
    raw["target"] = raw.target.replace({"facebook_page_category_top30": "facebook_page_reference"})
    raw = raw[raw.target.isin(TARGETS)].copy()
    rows = []
    for order in ORDERS:
        for rung in range(1, 10):
            model_id = model_for(order, rung)
            model_id = {
                "ss_ukr_rus": "ss_ukr_rus_twitter",
                "ss_covid": "ss_covid19_twitter",
                "ss_cp_hk": "ss_cp_hk_twitter",
            }.get(model_id, model_id)
            part = raw[raw.model_id.eq(model_id)]
            if set(part.target) != set(TARGETS) or len(part) != len(TARGETS):
                raise ValueError(f"SAMGPT {order}{rung}: incomplete common-target grid")
            for row in part.itertuples(index=False):
                rows.append(
                    {"order": order, "rung": rung, "target": row.target, "roc_auc": row.roc_auc_mean}
                )
    return decorate(pd.DataFrame(rows), "SAMGPT", 3, 500, "GraphCL")


def build_data() -> pd.DataFrame:
    frame = pd.concat([load_prodigy(), load_vision(), load_samgpt()], ignore_index=True)
    expected = {"PRODIGY": 135, "VISION": 75, "SAMGPT": 135}
    observed = frame.groupby("model").size().to_dict()
    if observed != expected:
        raise ValueError(f"unexpected ladder cell counts: {observed}")
    return frame


def plot(frame: pd.DataFrame) -> None:
    colors = dict(zip(TARGETS, plt.get_cmap("tab10").colors[: len(TARGETS)]))
    figure, axes = plt.subplots(
        len(MODEL_ROWS),
        len(ORDERS),
        figsize=(20.5, 12.2),
        sharey=True,
        constrained_layout=True,
    )
    values = frame.roc_auc.to_numpy()
    lower = max(0.45, np.floor((values.min() - 0.025) * 20) / 20)
    upper = min(1.0, np.ceil((values.max() + 0.025) * 20) / 20)

    for row_index, (model, detail) in enumerate(MODEL_ROWS):
        for column_index, (order, sources) in enumerate(ORDERS.items()):
            axis = axes[row_index, column_index]
            axis.set_ylim(lower, upper)
            axis.set_xlim(0.65, 9.35)
            axis.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
            axis.set_axisbelow(True)
            axis.spines[["top", "right"]].set_visible(False)
            if row_index == 0:
                axis.set_title(f"Order {order}", fontsize=14, fontweight="bold")

            part = frame[(frame.model == model) & (frame.order == order)]
            for target in TARGETS:
                target_rows = part[part.target.eq(target)].sort_values("rung")
                axis.plot(
                    target_rows.rung,
                    target_rows.roc_auc,
                    color=colors[target],
                    linewidth=1.2,
                    alpha=0.38,
                )
            mean = part.groupby("rung", as_index=False).roc_auc.mean().sort_values("rung")
            axis.plot(
                mean.rung,
                mean.roc_auc,
                color="#111111",
                linewidth=2.8,
                label="Five-target mean",
                zorder=6,
            )
            axis.set_xticks(
                range(1, 10),
                [f"{rung}\n{SOURCE_LABELS[source]}" for rung, source in enumerate(sources, start=1)],
                rotation=38,
                ha="right",
                rotation_mode="anchor",
                fontsize=8.2,
            )
            if column_index == 0:
                axis.set_ylabel(
                    f"{model}\n{detail}\n\nDownstream label-CLS ROC-AUC",
                    fontsize=10.5,
                )
            if row_index == len(MODEL_ROWS) - 1:
                axis.set_xlabel("Mixture size and graph added at this rung", fontsize=10)

    legend_handles = [
        plt.Line2D([0], [0], color=colors[target], linewidth=1.5, alpha=0.38, label=TARGET_LABELS[target])
        for target in TARGETS
    ]
    legend_handles.append(
        plt.Line2D([0], [0], color="#111111", linewidth=2.8, label="Five-target mean")
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 1.015),
        fontsize=9.5,
    )
    figure.suptitle(
        "Native-pretraining mixture ladders → downstream label classification",
        fontsize=17,
        fontweight="bold",
        y=1.045,
    )
    figure.text(
        0.5,
        -0.018,
        "Each colored line is one common labeled target; black is their unweighted mean. "
        "Every model uses its native pretext. VISION has only odd rungs; family-specific fixed-compute endpoints differ.",
        ha="center",
        fontsize=9.5,
    )
    output = ROOT / "figures/native_mixture_ladder_cls_grid"
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_order_mean_row(frame: pd.DataFrame) -> pd.DataFrame:
    order_mean = (
        frame.groupby(
            ["model", "rung", "target", "training_seeds", "checkpoint_step", "objective"],
            as_index=False,
        )
        .agg(roc_auc=("roc_auc", "mean"), orders_averaged=("order", "nunique"))
    )
    colors = dict(zip(TARGETS, plt.get_cmap("tab10").colors[: len(TARGETS)]))
    figure, axes = plt.subplots(1, len(MODEL_ROWS), figsize=(15.0, 5.4), sharey=True)
    values = order_mean.roc_auc.to_numpy()
    lower = max(0.45, np.floor((values.min() - 0.025) * 20) / 20)
    upper = min(1.0, np.ceil((values.max() + 0.025) * 20) / 20)

    for axis, (model, _detail) in zip(axes, MODEL_ROWS):
        part = order_mean[order_mean.model.eq(model)]
        for target in TARGETS:
            target_rows = part[part.target.eq(target)].sort_values("rung")
            axis.plot(
                target_rows.rung,
                target_rows.roc_auc,
                color=colors[target],
                linewidth=1.35,
                alpha=0.42,
            )
        overall = part.groupby("rung", as_index=False).roc_auc.mean().sort_values("rung")
        axis.plot(
            overall.rung,
            overall.roc_auc,
            color="#111111",
            linewidth=2.9,
            zorder=6,
        )
        axis.set_xlim(0.7, 9.3)
        axis.set_ylim(lower, upper)
        axis.set_xticks(range(1, 10))
        axis.set_xlabel("Number of pretraining graphs", fontsize=12)
        axis.set_title(model, fontsize=14.5, fontweight="bold", pad=7)
        axis.tick_params(axis="both", labelsize=11)
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Downstream label-CLS ROC-AUC", fontsize=12)
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color=colors[target],
            linewidth=1.5,
            alpha=0.42,
            label=TARGET_LABELS[target],
        )
        for target in TARGETS
    ]
    legend_handles.append(
        plt.Line2D([0], [0], color="#111111", linewidth=2.9, label="Five-target mean")
    )
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.905),
        fontsize=10.3,
    )
    figure.suptitle(
        "Native-pretraining mixture ladders: mean across Orders A/B/C",
        fontsize=18,
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.5,
        0.012,
        "Each colored curve averages a target across the three mixture orders; black additionally averages the five targets. "
        "VISION has only odd mixture sizes.",
        ha="center",
        fontsize=9.6,
        color="#666666",
    )
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 0.84), w_pad=0.35)
    output = ROOT / "figures/native_mixture_ladder_cls_order_mean_row"
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return order_mean


def plot_architecture_mean(order_mean: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    architecture_mean = (
        order_mean.groupby(["model", "rung"], as_index=False)
        .agg(
            roc_auc=("roc_auc", "mean"),
            targets_averaged=("target", "nunique"),
            orders_averaged=("orders_averaged", "min"),
        )
    )
    colors = {
        "PRODIGY": "#4e79a7",
        "VISION": "#59a14f",
        "SAMGPT": "#f28e2b",
    }
    single_source_means = (
        frame[frame.rung.eq(1)]
        .groupby(["model", "order"], as_index=False)
        .roc_auc.mean()
    )
    single_source_summary = (
        single_source_means.groupby("model", as_index=False)
        .agg(
            lowest_single=("roc_auc", "min"),
            mean_single=("roc_auc", "mean"),
            highest_single=("roc_auc", "max"),
        )
        .set_index("model")
    )
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 6.24), sharex=True, sharey=True)
    for axis, (model, _detail) in zip(axes, MODEL_ROWS):
        part = architecture_mean[architecture_mean.model.eq(model)].sort_values("rung")
        bounds = single_source_summary.loc[model]
        axis.plot(
            part.rung,
            part.roc_auc,
            color=colors[model],
            linewidth=3.5,
            zorder=3,
        )
        for value, linestyle, linewidth in (
            (bounds.lowest_single, "--", 1.7),
            (bounds.mean_single, "-", 2.0),
            (bounds.highest_single, "--", 1.7),
        ):
            axis.axhline(
                value,
                color="#777777",
                linewidth=linewidth,
                alpha=0.38,
                linestyle=linestyle,
                zorder=1,
            )
        axis.text(
            9.16,
            bounds.highest_single + 0.0015,
            f"High {bounds.highest_single:.3f}",
            ha="right",
            va="bottom",
            fontsize=11.8,
            color="#666666",
        )
        axis.text(
            9.16,
            bounds.mean_single,
            f"Mean {bounds.mean_single:.3f}",
            ha="right",
            va="center",
            fontsize=11.8,
            color="#555555",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
        )
        axis.text(
            9.16,
            bounds.lowest_single - 0.0015,
            f"Low {bounds.lowest_single:.3f}",
            ha="right",
            va="top",
            fontsize=11.8,
            color="#666666",
        )
        axis.set_title(model, fontsize=20, fontweight="bold", pad=7)
        axis.set_xlim(0.7, 9.3)
        axis.set_xticks(range(1, 10))
        axis.set_xlabel("Number of pretraining graphs", fontsize=14.5)
        axis.tick_params(axis="both", labelsize=13.5)
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.9)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    values = np.concatenate(
        [architecture_mean.roc_auc.to_numpy(), single_source_means.roc_auc.to_numpy()]
    )
    lower = np.floor((values.min() - 0.01) * 20) / 20
    upper = np.ceil((values.max() + 0.01) * 20) / 20
    axes[0].set_ylim(lower, upper)
    axes[0].set_ylabel("Mean downstream label-CLS ROC-AUC", fontsize=15.5)
    figure.suptitle(
        "Mixture scaling by architecture",
        fontsize=23,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.018,
        "Colored curves average Orders A/B/C and five common downstream targets. Gray lines show each architecture's low, mean, and high matched single-source means. VISION has only odd sizes.",
        ha="center",
        fontsize=11.8,
        color="#666666",
    )
    figure.tight_layout(rect=(0.0, 0.105, 1.0, 0.91), w_pad=0.35)
    output = ROOT / "figures/native_mixture_ladder_cls_architecture_mean"
    figure.savefig(output.with_suffix(".png"), dpi=480, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    return architecture_mean


def main() -> None:
    frame = build_data()
    frame.to_csv(ROOT / "data/native_mixture_ladder_cls_grid.csv", index=False)
    plot(frame)
    order_mean = plot_order_mean_row(frame)
    order_mean.to_csv(ROOT / "data/native_mixture_ladder_cls_order_mean.csv", index=False)
    architecture_mean = plot_architecture_mean(order_mean, frame)
    architecture_mean.to_csv(
        ROOT / "data/native_mixture_ladder_cls_architecture_mean.csv", index=False
    )
    print(
        "NATIVE_MIXTURE_LADDER_GRID_OK "
        f"rows={len(frame)} models={frame.model.nunique()} "
        f"orders={sorted(frame.order.unique())} targets={frame.target.nunique()} "
        f"order_mean_rows={len(order_mean)} architecture_mean_rows={len(architecture_mean)}"
    )


if __name__ == "__main__":
    main()
