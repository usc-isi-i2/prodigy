#!/usr/bin/env python3
"""Plot five-target mean label-CLS AUC for every native source configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[5]
FINAL_CORE_SETUP = REPO / "scripts/experiments/setup/final_core"
sys.path.insert(0, str(FINAL_CORE_SETUP))
from core_plan import ORDERS  # noqa: E402


TARGETS = (
    "covid_political",
    "election2020",
    "facebook_page_reference",
    "twibot20",
    "ukr_rus_suspended",
)
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
MODEL_META = {
    "PRODIGY": ("neighbor matching", "matched 128-episode ladder · step 2,500"),
    "VISION": ("feature similarity", "single sources: step 900 · mixtures: step 2,500"),
    "SAMGPT": ("GraphCL", "single sources and mixtures: step 500"),
}
SINGLE_COLOR = "#4e79a7"
MIXTURE_COLOR = "#59a14f"


def source_key(sources: tuple[str, ...]) -> str:
    return "|".join(sorted(SOURCE_CANONICAL[source] for source in sources))


def load_singles() -> pd.DataFrame:
    path = ROOT / "data/native_specialists_vs_all9_cls.csv"
    raw = pd.read_csv(path)
    raw = raw[
        raw.kind.eq("single source")
        & raw.target.isin(TARGETS)
        & raw.model.isin(MODEL_META)
        & raw.model.ne("PRODIGY")
    ].copy()
    raw["canonical_source"] = raw.source.map(SOURCE_CANONICAL)
    if raw.canonical_source.isna().any():
        raise ValueError(f"unknown single sources: {raw[raw.canonical_source.isna()].source.unique()}")
    result = (
        raw.groupby(["model", "canonical_source", "checkpoint_step"], as_index=False)
        .agg(mean_roc_auc=("roc_auc", "mean"), target_count=("target", "nunique"))
    )
    result["config_type"] = "single source"
    result["config_id"] = "ss:" + result.canonical_source
    result["label"] = result.canonical_source.map(SOURCE_LABELS)
    result["n_sources"] = 1
    result["order"] = ""
    result["rung"] = 1
    result["sources"] = result.canonical_source
    return result.drop(columns="canonical_source")


def load_prodigy_matched_singles() -> pd.DataFrame:
    path = (
        REPO
        / "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/"
        "data/classification_ladder/classification_long.tsv"
    )
    raw = pd.read_csv(path, sep="\t")
    raw = raw[raw.checkpoint_step.eq(2500) & raw.dataset.isin(TARGETS)].copy()
    rows = []
    for order, sources in ORDERS.items():
        source = sources[0]
        model_id = f"ss_{source}"
        part = raw[raw.model_id.eq(model_id)]
        if (
            len(part) != 3 * len(TARGETS)
            or set(part.training_seed) != {0, 1, 2}
            or set(part.dataset) != set(TARGETS)
        ):
            raise ValueError(f"PRODIGY matched specialist {model_id}: incomplete ladder grid")
        canonical_source = SOURCE_CANONICAL[source]
        rows.append(
            {
                "model": "PRODIGY",
                "mean_roc_auc": part.roc_auc.mean(),
                "target_count": part.dataset.nunique(),
                "checkpoint_step": 2500,
                "config_type": "single source",
                "config_id": f"ss:{canonical_source}",
                "label": SOURCE_LABELS[canonical_source],
                "n_sources": 1,
                "order": order,
                "rung": 1,
                "sources": canonical_source,
            }
        )
    return pd.DataFrame(rows)


def load_mixtures() -> pd.DataFrame:
    path = ROOT / "data/native_mixture_ladder_cls_grid.csv"
    raw = pd.read_csv(path)
    raw = raw[
        raw.model.isin(MODEL_META)
        & raw.target.isin(TARGETS)
        & raw.rung.gt(1)
    ].copy()
    raw["source_tuple"] = [tuple(ORDERS[order][: int(rung)]) for order, rung in zip(raw.order, raw.rung)]
    raw["config_id"] = [source_key(sources) for sources in raw.source_tuple]
    raw["sources"] = [",".join(SOURCE_CANONICAL[source] for source in sources) for sources in raw.source_tuple]
    raw["n_sources"] = raw.rung.astype(int)

    per_target = (
        raw.groupby(["model", "config_id", "target"], as_index=False)
        .agg(roc_auc=("roc_auc", "mean"))
    )
    meta = (
        raw.sort_values(["rung", "order"])
        .groupby(["model", "config_id"], as_index=False)
        .first()[["model", "config_id", "order", "rung", "sources", "n_sources", "checkpoint_step"]]
    )
    result = (
        per_target.groupby(["model", "config_id"], as_index=False)
        .agg(mean_roc_auc=("roc_auc", "mean"), target_count=("target", "nunique"))
        .merge(meta, on=["model", "config_id"], validate="one_to_one")
    )
    result["config_type"] = "mixture"
    result["label"] = [f"{int(rung)} graphs" for rung in result.rung]
    return result


def build_data() -> pd.DataFrame:
    frame = pd.concat(
        [load_singles(), load_prodigy_matched_singles(), load_mixtures()],
        ignore_index=True,
    )
    frame = frame[
        frame.config_type.eq("single source")
        | (frame.config_type.eq("mixture") & frame.order.eq("A"))
    ].copy()
    if not frame.target_count.eq(len(TARGETS)).all():
        bad = frame[~frame.target_count.eq(len(TARGETS))]
        raise ValueError(f"incomplete target means:\n{bad}")
    objective = {model: meta[0] for model, meta in MODEL_META.items()}
    frame["objective"] = frame.model.map(objective)
    return frame[
        [
            "model",
            "objective",
            "config_type",
            "config_id",
            "label",
            "n_sources",
            "order",
            "rung",
            "sources",
            "checkpoint_step",
            "target_count",
            "mean_roc_auc",
        ]
    ]


def plot_model(frame: pd.DataFrame, model: str) -> None:
    part = frame[frame.model.eq(model)].copy()
    part["single_rank"] = part.apply(
        lambda row: SOURCE_ORDER.index(row.config_id.removeprefix("ss:"))
        if row.config_type == "single source"
        else 99,
        axis=1,
    )
    part["type_rank"] = part.config_type.map({"single source": 0, "mixture": 1})
    part = part.sort_values(["type_rank", "single_rank", "n_sources", "order", "label"])

    width = max(9.5, 2.8 + 0.62 * len(part))
    figure, axis = plt.subplots(figsize=(width, 6.4))
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.86,
        bottom=0.29,
    )
    colors = part.config_type.map({"single source": SINGLE_COLOR, "mixture": MIXTURE_COLOR})
    bars = axis.bar(
        range(len(part)),
        part.mean_roc_auc - 0.45,
        bottom=0.45,
        color=colors,
        edgecolor="white",
        linewidth=0.55,
        width=0.76,
    )
    axis.set_xticks(range(len(part)), part.label, rotation=52, ha="right", rotation_mode="anchor", fontsize=9)
    axis.set_ylim(0.45, 1.0)
    axis.set_yticks([0.45, *[value / 20 for value in range(10, 21)]])
    axis.set_ylabel("Mean downstream label-classification ROC-AUC\nacross five targets")
    axis.set_xlabel("Native pretraining source configuration (Order A mixtures)")
    objective, compute_note = MODEL_META[model]
    axis.set_title(
        f"{model}: single sources and Order A mixtures\n{objective} · {compute_note}",
        fontsize=14,
        fontweight="bold",
    )
    axis.axhline(0.5, color="#888888", linestyle="--", linewidth=0.9, alpha=0.8)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)

    for bar, value in zip(bars, part.mean_roc_auc):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.008, 0.988),
            f"{value:.3f}",
            va="bottom" if value < 0.975 else "top",
            ha="center",
            fontsize=8.3,
            rotation=90,
        )

    n_singles = int(part.config_type.eq("single source").sum())
    axis.axvline(n_singles - 0.5, color="#777777", linewidth=0.9)
    axis.legend(
        handles=[
            Patch(facecolor=SINGLE_COLOR, label="Single source"),
            Patch(facecolor=MIXTURE_COLOR, label="Mixture"),
        ],
        frameon=False,
        loc="upper right",
        ncol=2,
    )
    stem = ROOT / "figures" / f"{model.lower()}_native_config_mean_cls_bars"
    figure.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    frame = build_data()
    frame.to_csv(ROOT / "data/native_config_mean_cls_bars.csv", index=False)
    for model in MODEL_META:
        plot_model(frame, model)
    counts = frame.groupby(["model", "config_type"]).size().to_dict()
    print(f"NATIVE_CONFIG_MEAN_BARS_OK rows={len(frame)} counts={counts}")


if __name__ == "__main__":
    main()
