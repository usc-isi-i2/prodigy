#!/usr/bin/env python3
"""Plot native single-source CLS transfer against the all-nine mixture."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[5]
TARGETS = (
    "covid_political",
    "election2020",
    "facebook_page_reference",
    "twibot20",
    "ukr_rus_suspended",
)
TARGET_LABELS = {
    "covid_political": "COVID\npolitical",
    "election2020": "Election2020",
    "facebook_page_reference": "Facebook\npages",
    "twibot20": "TwiBot20",
    "ukr_rus_suspended": "UKR/RUS\nsuspended",
    "mean": "Mean",
}
MODEL_META = {
    "PRODIGY": ("neighbor matching", 2500, 3),
    "VISION": ("feature similarity", 900, 1),
    "SAMGPT": ("GraphCL", 500, 3),
}
SOURCE_CANONICAL = {
    "covid": "covid_twitter",
    "covid19_twitter": "covid_twitter",
    "covid_political": "covid_political",
    "cp_hk": "cp_hk_twitter",
    "cp_hk_twitter": "cp_hk_twitter",
    "election2020": "election2020",
    "facebook_page_reference": "facebook_page_reference",
    "midterm": "midterm",
    "twibot20": "twibot20",
    "ukr_rus": "ukr_rus_twitter",
    "ukr_rus_twitter": "ukr_rus_twitter",
    "ukr_rus_suspended": "ukr_rus_suspended",
}
SOURCE_ORDER = (
    "covid_twitter",
    "covid_political",
    "cp_hk_twitter",
    "election2020",
    "facebook_page_reference",
    "midterm",
    "twibot20",
    "ukr_rus_twitter",
    "ukr_rus_suspended",
)
SOURCE_LABELS = {
    "covid_twitter": "COVID Twitter",
    "covid_political": "COVID political",
    "cp_hk_twitter": "CP/HK Twitter",
    "election2020": "Election2020",
    "facebook_page_reference": "Facebook pages",
    "midterm": "Midterm",
    "twibot20": "TwiBot20",
    "ukr_rus_twitter": "UKR/RUS Twitter",
    "ukr_rus_suspended": "UKR/RUS suspended",
}
SOURCE_COLORS = dict(
    zip(
        SOURCE_ORDER,
        (
            "#4e79a7",
            "#76b7b2",
            "#59a14f",
            "#e15759",
            "#b07aa1",
            "#ff9da7",
            "#9c755f",
            "#bab0ac",
            "#edc948",
        ),
    )
)


def load_prodigy() -> pd.DataFrame:
    specialist_path = (
        REPO
        / "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/"
        "data/prodigy_final_core/auc/summary/single_source_metrics_long.tsv"
    )
    specialists = pd.read_csv(specialist_path, sep="\t")
    specialists = specialists[
        specialists.target.isin(TARGETS) & specialists.checkpoint_step.eq(2500)
    ]
    specialists = (
        specialists.groupby(["source", "target"], as_index=False)
        .agg(roc_auc=("roc_auc_ovr_macro", "mean"), training_seeds=("seed", "nunique"))
        .assign(model="PRODIGY", kind="single source", checkpoint_step=2500)
    )

    mixture_path = (
        REPO
        / "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/"
        "data/classification_ladder/classification_long.tsv"
    )
    mixture = pd.read_csv(mixture_path, sep="\t")
    mixture = mixture[
        mixture.model_id.eq("all9")
        & mixture.dataset.isin(TARGETS)
        & mixture.checkpoint_step.eq(2500)
    ]
    mixture = (
        mixture.groupby("dataset", as_index=False)
        .agg(roc_auc=("roc_auc", "mean"), training_seeds=("training_seed", "nunique"))
        .rename(columns={"dataset": "target"})
        .assign(model="PRODIGY", kind="all-nine mixture", source="all9", checkpoint_step=2500)
    )
    return pd.concat([specialists, mixture], ignore_index=True)


def load_vision() -> pd.DataFrame:
    specialist_path = ROOT / "data/cross_graph_cls_social_sources_9x5.csv"
    specialists = pd.read_csv(specialist_path)
    specialists = specialists[
        specialists.model.eq("VISION")
        & specialists.available
        & specialists.target.isin(TARGETS)
        & specialists.checkpoint_step.eq(900)
    ][["source", "target", "roc_auc", "seeds"]].rename(columns={"seeds": "training_seeds"})
    specialists = specialists.assign(
        model="VISION", kind="single source", checkpoint_step=900
    )

    rows = []
    raw_root = ROOT / "data/vision_all9_saturation_raw"
    for seed in range(3):
        path = raw_root / f"vision_all9_s{seed}_step900_cls.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["dataset"] in TARGETS:
                    rows.append(row)
    mixture = pd.DataFrame(rows)
    mixture = (
        mixture.groupby("dataset", as_index=False)
        .agg(roc_auc=("roc_auc", "mean"), training_seeds=("training_seed", "nunique"))
        .rename(columns={"dataset": "target"})
        .assign(model="VISION", kind="all-nine mixture", source="all9", checkpoint_step=900)
    )
    return pd.concat([specialists, mixture], ignore_index=True)


def load_samgpt() -> pd.DataFrame:
    path = (
        REPO
        / "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/"
        "data/samgpt_downstream_cls/three_seed_mean.csv"
    )
    raw = pd.read_csv(path)
    raw["target"] = raw.target.replace(
        {"facebook_page_category_top30": "facebook_page_reference"}
    )
    raw = raw[raw.target.isin(TARGETS)]
    specialists = raw[raw.n_sources.eq(1)][
        ["sources", "target", "roc_auc_mean", "training_seeds"]
    ].rename(columns={"sources": "source", "roc_auc_mean": "roc_auc"})
    specialists = specialists.assign(
        model="SAMGPT", kind="single source", checkpoint_step=500
    )
    mixture = raw[raw.n_sources.eq(9)][
        ["target", "roc_auc_mean", "training_seeds"]
    ].rename(columns={"roc_auc_mean": "roc_auc"})
    mixture = mixture.assign(
        model="SAMGPT", kind="all-nine mixture", source="all9", checkpoint_step=500
    )
    return pd.concat([specialists, mixture], ignore_index=True)


def add_means(frame: pd.DataFrame) -> pd.DataFrame:
    means = (
        frame.groupby(["model", "kind", "source", "training_seeds", "checkpoint_step"], as_index=False)
        .roc_auc.mean()
        .assign(target="mean")
    )
    return pd.concat([frame, means], ignore_index=True)


def build_data() -> pd.DataFrame:
    frame = pd.concat([load_prodigy(), load_vision(), load_samgpt()], ignore_index=True)
    for model in MODEL_META:
        part = frame[frame.model.eq(model)]
        mixture = part[part.kind.eq("all-nine mixture")]
        if len(mixture) != len(TARGETS) or set(mixture.target) != set(TARGETS):
            raise ValueError(f"{model}: incomplete mixture target grid")
        specialists = part[part.kind.eq("single source")]
        counts = specialists.groupby("source").target.nunique()
        if counts.empty or not counts.eq(len(TARGETS)).all():
            raise ValueError(f"{model}: incomplete single-source rows {counts.to_dict()}")
    return add_means(frame)


def plot_model(frame: pd.DataFrame, model: str) -> None:
    part = frame[frame.model.eq(model)]
    specialists = part[part.kind.eq("single source")]
    mixture = part[part.kind.eq("all-nine mixture")]
    sources = sorted(
        specialists.source.unique(),
        key=lambda source: SOURCE_ORDER.index(SOURCE_CANONICAL[source]),
    )
    offsets = dict(zip(sources, np.linspace(-0.045, 0.045, len(sources))))
    ticks = (*TARGETS, "mean")
    x_by_target = {target: index for index, target in enumerate(ticks)}

    figure, axis = plt.subplots(figsize=(9.4, 5.6), constrained_layout=True)
    for source in sources:
        canonical_source = SOURCE_CANONICAL[source]
        source_rows = specialists[specialists.source.eq(source)].set_index("target")
        axis.scatter(
            [x_by_target[target] + offsets[source] for target in ticks],
            [source_rows.loc[target, "roc_auc"] for target in ticks],
            s=38,
            color=SOURCE_COLORS[canonical_source],
            edgecolor="white",
            linewidth=0.55,
            alpha=0.82,
            zorder=3,
        )
    mixture_rows = mixture.set_index("target")
    axis.scatter(
        range(len(ticks)),
        [mixture_rows.loc[target, "roc_auc"] for target in ticks],
        marker="*",
        s=190,
        color="#111111",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="All-nine mixture",
    )
    axis.axvline(len(ticks) - 1.5, color="#bdbdbd", linewidth=1.0, linestyle="--")
    axis.set_xticks(range(len(ticks)), [TARGET_LABELS[target] for target in ticks])
    axis.set_ylabel("Downstream label-classification ROC-AUC")
    axis.set_xlabel("Downstream target graph")
    objective, step, seeds = MODEL_META[model]
    axis.set_title(
        f"{model}: native specialists versus all-nine mixture\n"
        f"{objective} · step {step:,} · {seeds} training seed{'s' if seeds != 1 else ''}",
        fontsize=14,
        fontweight="bold",
    )
    all_values = part.roc_auc.to_numpy()
    low = max(0.45, np.floor((all_values.min() - 0.025) * 20) / 20)
    high = min(1.0, np.ceil((all_values.max() + 0.025) * 20) / 20)
    axis.set_ylim(low, high)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    source_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=6,
            markerfacecolor=SOURCE_COLORS[SOURCE_CANONICAL[source]],
            markeredgecolor="white",
            label=SOURCE_LABELS[SOURCE_CANONICAL[source]],
        )
        for source in sources
    ]
    mixture_handle = Line2D(
        [],
        [],
        marker="*",
        linestyle="none",
        markersize=11,
        markerfacecolor="#111111",
        markeredgecolor="white",
        label="All-nine mixture",
    )
    axis.legend(
        handles=[*source_handles, mixture_handle],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=5,
        fontsize=8.5,
        handletextpad=0.35,
        columnspacing=1.15,
    )
    axis.text(
        0.0,
        -0.31,
        f"{len(sources)} available native single-source models; jitter is visual only.",
        transform=axis.transAxes,
        fontsize=9,
        color="#666666",
    )
    stem = ROOT / "figures" / f"{model.lower()}_specialists_vs_all9_cls"
    figure.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_model_row(frame: pd.DataFrame) -> None:
    ticks = (*TARGETS, "mean")
    x_by_target = {target: index for index, target in enumerate(ticks)}
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.4), sharey=True)

    for axis, model in zip(axes, MODEL_META):
        part = frame[frame.model.eq(model)]
        specialists = part[part.kind.eq("single source")]
        mixture = part[part.kind.eq("all-nine mixture")]
        sources = sorted(
            specialists.source.unique(),
            key=lambda source: SOURCE_ORDER.index(SOURCE_CANONICAL[source]),
        )
        offsets = dict(zip(sources, np.linspace(-0.045, 0.045, len(sources))))

        for source in sources:
            canonical_source = SOURCE_CANONICAL[source]
            source_rows = specialists[specialists.source.eq(source)].set_index("target")
            axis.scatter(
                [x_by_target[target] + offsets[source] for target in ticks],
                [source_rows.loc[target, "roc_auc"] for target in ticks],
                s=32,
                color=SOURCE_COLORS[canonical_source],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.82,
                zorder=3,
            )

        mixture_rows = mixture.set_index("target")
        axis.scatter(
            range(len(ticks)),
            [mixture_rows.loc[target, "roc_auc"] for target in ticks],
            marker="*",
            s=155,
            color="#111111",
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        axis.axvline(len(ticks) - 1.5, color="#bdbdbd", linewidth=0.9, linestyle="--")
        axis.set_xticks(
            range(len(ticks)),
            [TARGET_LABELS[target] for target in ticks],
            fontsize=8.5,
        )
        objective, step, seeds = MODEL_META[model]
        axis.set_title(
            f"{model}\n{objective} · step {step:,} · {seeds} seed{'s' if seeds != 1 else ''}",
            fontsize=12,
            fontweight="bold",
        )
        axis.set_ylim(0.45, 1.0)
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Downstream label-classification ROC-AUC")
    figure.suptitle(
        "Native single-source specialists versus all-nine mixture",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    figure.supxlabel("Downstream target graph", y=0.205)

    source_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=6,
            markerfacecolor=SOURCE_COLORS[source],
            markeredgecolor="white",
            label=SOURCE_LABELS[source],
        )
        for source in SOURCE_ORDER
    ]
    mixture_handle = Line2D(
        [],
        [],
        marker="*",
        linestyle="none",
        markersize=11,
        markerfacecolor="#111111",
        markeredgecolor="white",
        label="All-nine mixture",
    )
    figure.legend(
        handles=[*source_handles, mixture_handle],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=5,
        fontsize=9,
        handletextpad=0.35,
        columnspacing=1.25,
    )
    figure.text(
        0.5,
        0.012,
        "Jitter is visual only. VISION includes the six currently available native single-source models.",
        ha="center",
        fontsize=8.5,
        color="#666666",
    )
    figure.tight_layout(rect=(0.0, 0.23, 1.0, 0.94), w_pad=1.2)
    stem = ROOT / "figures/native_specialists_vs_all9_cls_row"
    figure.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    frame = build_data()
    frame.to_csv(ROOT / "data/native_specialists_vs_all9_cls.csv", index=False)
    for model in MODEL_META:
        plot_model(frame, model)
    plot_model_row(frame)
    print(
        "SPECIALISTS_VS_MIXTURE_CLS_OK "
        f"rows={len(frame)} source_counts="
        f"{frame[frame.kind.eq('single source')].groupby('model').source.nunique().to_dict()}"
    )


if __name__ == "__main__":
    main()
