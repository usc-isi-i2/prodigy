#!/usr/bin/env python3
"""Plot constituent-specialist max-rule diagnostics for four ladder/task pairs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[6]
FINAL_DATA = HERE / "data/prodigy_final_core"
PLAN = FINAL_DATA / "training/plan.tsv"
FIXED_COMPUTE_NM = FINAL_DATA / "log_recovered_metrics/physical_metrics.tsv"
FIXED_COMPUTE_NM_SS = FINAL_DATA / "auc/summary/single_source_metrics_long.tsv"
FIXED_COMPUTE_CLS = HERE / "data/classification_ladder/classification_long.tsv"
FINAL_CLS_SS = (
    REPO / "scripts/experiments/analysis/transfer/matrices/cross_architecture/icl_arch_matrix/"
    "data/finalcore_cls2500_seed2/classification_auc.tsv"
)
HISTORICAL_NM_SS = (
    REPO / "scripts/experiments/analysis/transfer/matrices/prodigy_nm/single_source/"
    "nm_single_source_matrix_facebook/data/nm_single_source_matrix_9x9_long.csv"
)
HISTORICAL_CLS_SS = (
    REPO / "scripts/experiments/analysis/transfer/matrices/prodigy_nm/downstream/"
    "nm_single_source_downstream/data/classification.csv"
)
FIXED_EXPOSURE_NM = (
    REPO / "scripts/experiments/analysis/transfer/ablations/prodigy_nm/source_exposure/"
    "nm_ladder_fixed_exposure_nhop2/data/logical_results.csv"
)
FIXED_EXPOSURE_CLS = (
    REPO / "scripts/experiments/analysis/transfer/ablations/prodigy_nm/downstream/"
    "nm_ladder_downstream_nhop2/data/classification_roc_auc.csv"
)
FIXED_EXPOSURE_FIGURES = FIXED_EXPOSURE_CLS.parent.parent / "figures"

TARGETS = ("covid_political", "election2020", "twibot20", "ukr_rus_suspended")
TARGET_LABELS = {
    "covid_political": "COVID political",
    "election2020": "Election 2020",
    "twibot20": "TwiBot-20",
    "ukr_rus_suspended": "UKR–RUS suspended",
}
TARGET_MARKERS = {
    "covid_political": "o",
    "election2020": "s",
    "twibot20": "^",
    "ukr_rus_suspended": "D",
}
ORDER_COLORS = {"A": "#0072B2", "B": "#D55E00", "C": "#009E73"}
ALIASES = {
    "ukr_rus_twitter": "ukr_rus",
    "covid19_twitter": "covid",
    "cp_hk_twitter": "cp_hk",
}


def normalize_source(value: str) -> str:
    return ALIASES.get(value, value)


def read_plan() -> pd.DataFrame:
    rows = []
    for row in pd.read_csv(PLAN, sep="\t").itertuples(index=False):
        sources = tuple(row.sources.split(","))
        for alias in row.aliases.split(","):
            if not alias.startswith("ladder:"):
                continue
            _, order, rung = alias.split(":")
            rows.append({"model_id": row.model_id, "order": order, "rung": int(rung), "sources": sources})
    return pd.DataFrame(rows)


def historical_classification_specialists() -> dict[tuple[str, str], float]:
    data = pd.read_csv(HISTORICAL_CLS_SS)
    values = {
        (normalize_source(row.source), target): float(getattr(row, target))
        for row in data.itertuples(index=False)
        for target in TARGETS
    }
    facebook = pd.read_csv(FINAL_CLS_SS, sep="\t")
    facebook = facebook[facebook.sources.eq("facebook_page_reference")]
    for row in facebook.itertuples(index=False):
        if row.dataset in TARGETS:
            values[("facebook_page_reference", row.dataset)] = float(row.roc_auc)
    return values


def historical_nm_specialists() -> dict[tuple[str, str], float]:
    data = pd.read_csv(HISTORICAL_NM_SS)
    data = data[data.metric.eq("roc_auc")]
    return {
        (normalize_source(row.train), normalize_source(row.test)): float(row.value)
        for row in data.itertuples(index=False)
    }


def fixed_compute_nm_points(rank: int = 1) -> pd.DataFrame:
    plan = read_plan()
    ladder = pd.read_csv(FIXED_COMPUTE_NM, sep="\t")
    ladder = ladder[ladder.target.isin(TARGETS)].groupby(["model_id", "target"], as_index=False).roc_auc_ovr_macro_logged.mean()
    specialists = pd.read_csv(FIXED_COMPUTE_NM_SS, sep="\t")
    specialists = specialists[specialists.target.isin(TARGETS)].groupby(["source", "target"]).roc_auc_ovr_macro.mean().to_dict()
    return build_points(plan, ladder, "target", "roc_auc_ovr_macro_logged", specialists, rank)


def fixed_compute_cls_points(rank: int = 1) -> pd.DataFrame:
    plan = read_plan()
    ladder = pd.read_csv(FIXED_COMPUTE_CLS, sep="\t")
    ladder = ladder[ladder.dataset.isin(TARGETS)].groupby(["model_id", "dataset"], as_index=False).roc_auc.mean()
    return build_points(plan, ladder, "dataset", "roc_auc", historical_classification_specialists(), rank)


def build_points(
    plan: pd.DataFrame,
    ladder: pd.DataFrame,
    target_column: str,
    value_column: str,
    specialists: dict[tuple[str, str], float],
    rank: int,
) -> pd.DataFrame:
    merged = plan.merge(ladder, on="model_id", validate="many_to_many")
    rows = []
    for row in merged.itertuples(index=False):
        if row.rung < 2:
            continue
        target = getattr(row, target_column)
        constituent_scores = sorted(specialists[(source, target)] for source in row.sources)
        prediction = constituent_scores[-rank]
        rows.append({"order": row.order, "rung": row.rung, "target": target, "x": prediction, "y": getattr(row, value_column)})
    return pd.DataFrame(rows)


def fixed_exposure_points(task: str, rank: int = 1) -> pd.DataFrame:
    specialists = historical_nm_specialists() if task == "nm" else historical_classification_specialists()
    if task == "nm":
        data = pd.read_csv(FIXED_EXPOSURE_NM)
        data = data[data.dataset.isin(TARGETS)].copy()
        data["target"] = data.dataset
        data["y"] = data.test_roc_auc
    else:
        wide = pd.read_csv(FIXED_EXPOSURE_CLS)
        data = wide[wide.variant.eq("fixed10k")].melt(
            id_vars=["order", "rung", "sources"], value_vars=list(TARGETS),
            var_name="target", value_name="y",
        )
    rows = []
    for row in data.itertuples(index=False):
        if row.rung < 2:
            continue
        sources = tuple(normalize_source(source) for source in row.sources.split(","))
        constituent_scores = sorted(specialists[(source, row.target)] for source in sources)
        prediction = constituent_scores[-rank]
        rows.append({"order": row.order, "rung": row.rung, "target": row.target, "x": prediction, "y": row.y})
    return pd.DataFrame(rows)


def plot(points: pd.DataFrame, title: str, output: Path, note: str, predictor_label: str) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    for (order, target), part in points.groupby(["order", "target"]):
        ax.scatter(
            part.x, part.y, s=48, marker=TARGET_MARKERS[target],
            facecolor=ORDER_COLORS[order], edgecolor="white", linewidth=.55, alpha=.82,
        )
    low = min(points.x.min(), points.y.min()) - .025
    high = max(points.x.max(), points.y.max()) + .025
    low, high = max(.4, low), min(1.0, high)
    ax.plot([low, high], [low, high], color="#555555", ls="--", lw=1.2, label=predictor_label)
    ax.set(xlim=(low, high), ylim=(low, high), xlabel=f"AUC of {predictor_label} constituent specialist", ylabel="AUC of mixture")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#dddddd", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    error = points.y - points.x
    corr = np.corrcoef(points.x, points.y)[0, 1]
    ax.text(
        .03, .97,
        f"n = {len(points)}  ·  MAE = {error.abs().mean():.3f}  ·  r = {corr:.3f}\n"
        f"mixture below reference: {(error < 0).mean():.0%}",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .82, "pad": 3},
    )
    order_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=8, label=f"Order {order}")
        for order, color in ORDER_COLORS.items() if order in set(points.order)
    ]
    target_handles = [
        Line2D([0], [0], marker=TARGET_MARKERS[target], color="#555555", linestyle="none", markersize=7, label=TARGET_LABELS[target])
        for target in TARGETS
    ]
    ax.legend(handles=order_handles + target_handles, frameon=False, fontsize=8, ncol=2, loc="lower right")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=.98)
    fig.text(.5, .015, note, ha="center", fontsize=8, color="#666666")
    fig.tight_layout(rect=(0, .035, 1, .95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    jobs = []
    for rank, rule_name, stem in ((1, "best", "max_rule"), (2, "second-best", "second_best_rule")):
        jobs.extend([
            (fixed_compute_nm_points(rank), f"Fixed compute · NM {rule_name} rule", HERE / f"figures/pngs/{stem}_fixed_compute_nm", "2,500 updates total · 3-seed means · matched final-core specialist matrix", rule_name),
            (fixed_compute_cls_points(rank), f"Fixed compute · classification {rule_name} rule", HERE / f"figures/pngs/{stem}_fixed_compute_classification", "2,500-update mixtures · 3-seed means · matrix-reference specialists", rule_name),
            (fixed_exposure_points("nm", rank), f"Fixed exposure · NM {rule_name} rule", FIXED_EXPOSURE_FIGURES / f"{stem}_fixed_exposure_nm", "10k updates/source mixtures · 1 seed · historical specialist-matrix reference", rule_name),
            (fixed_exposure_points("classification", rank), f"Fixed exposure · classification {rule_name} rule", FIXED_EXPOSURE_FIGURES / f"{stem}_fixed_exposure_classification", "10k updates/source mixtures · 1 seed · historical specialist-matrix reference", rule_name),
        ])
    for points, title, output, note, predictor_label in jobs:
        plot(points, title, output, note, predictor_label)
        print(output.with_suffix(".png"))


if __name__ == "__main__":
    main()
