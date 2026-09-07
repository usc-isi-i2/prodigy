#!/usr/bin/env python3
"""Plot native-specialist cross-graph downstream-CLS matrices on common targets."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[5]
COMMON_GRAPHS = (
    "covid_political",
    "election2020",
    "twibot20",
    "ukr_rus_suspended",
)
SOCIAL_TARGETS = (
    "covid_political",
    "election2020",
    "facebook_page_reference",
    "twibot20",
    "ukr_rus_suspended",
)
PLOT_TARGETS = (*SOCIAL_TARGETS, "mean")
SOCIAL_SOURCES = (
    "covid",
    "cp_hk",
    "midterm",
    "ukr_rus",
    # Keep the five source/target graphs together at the bottom of the matrix.
    "covid_political",
    "election2020",
    "facebook_page_reference",
    "twibot20",
    "ukr_rus_suspended",
)
PLOT_SOURCES = (*SOCIAL_SOURCES, "all9")
DISPLAY = {
    "covid": "COVID\nretweet*",
    "covid_political": "COVID\npolitical",
    "cp_hk": "COVID protest\nHK*",
    "election2020": "Election\n2020",
    "facebook_page_reference": "Facebook\npages",
    "midterm": "US\nmidterm*",
    "twibot20": "TwiBot-20",
    "ukr_rus": "Ukraine/\nRussia*",
    "ukr_rus_suspended": "UKR/RUS\nsuspended",
    "all9": "Final mixture\n(all nine)",
    "mean": "Mean",
}


def _complete(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    expected = {(source, target) for source in COMMON_GRAPHS for target in COMMON_GRAPHS}
    observed = set(zip(frame["source"], frame["target"]))
    if observed != expected or len(frame) != len(expected):
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{model}: expected one complete 4x4 matrix; "
            f"rows={len(frame)} missing={missing} extra={extra}"
        )
    return frame


def load_prodigy() -> pd.DataFrame:
    path = (
        REPO
        / "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/"
        "data/prodigy_final_core/auc/summary/single_source_metrics_long.tsv"
    )
    frame = pd.read_csv(path, sep="\t")
    frame = frame[
        frame["source"].isin(COMMON_GRAPHS)
        & frame["target"].isin(COMMON_GRAPHS)
        & frame["checkpoint_step"].eq(2500)
    ]
    grouped = (
        frame.groupby(["source", "target"], as_index=False)
        .agg(
            roc_auc=("roc_auc_ovr_macro", "mean"),
            sample_std=("roc_auc_ovr_macro", "std"),
            seeds=("seed", "nunique"),
        )
        .assign(
            model="PRODIGY",
            objective="neighbor matching",
            checkpoint_step=2500,
            protocol="128 fixed 10-shot episodes",
        )
    )
    return _complete(grouped, "PRODIGY")


def load_vision() -> pd.DataFrame:
    path = (
        REPO
        / "scripts/experiments/analysis/transfer/matrices/cross_architecture/"
        "icl_arch_matrix/data/native_source_900_seed0/classification_all.tsv"
    )
    frame = pd.read_csv(path, sep="\t")
    frame = frame[
        frame["architecture"].eq("vision")
        & frame["checkpoint_step"].eq(900)
        & frame["dataset"].isin(COMMON_GRAPHS)
    ].copy()
    frame["source_list"] = frame["sources"].map(ast.literal_eval)
    frame = frame[frame["source_list"].map(len).eq(1)]
    frame["source"] = frame["source_list"].str[0]
    frame = frame[frame["source"].isin(COMMON_GRAPHS)]
    grouped = (
        frame.rename(columns={"dataset": "target"})
        .groupby(["source", "target"], as_index=False)
        .agg(
            roc_auc=("roc_auc", "mean"),
            sample_std=("roc_auc", lambda _: np.nan),
            seeds=("training_seed", "nunique"),
        )
        .assign(
            model="VISION",
            objective="feature similarity",
            checkpoint_step=900,
            protocol="128 fixed 10-shot episodes",
        )
    )
    return _complete(grouped, "VISION")


def load_samgpt() -> pd.DataFrame:
    path = (
        REPO
        / "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/"
        "data/samgpt_downstream_cls/three_seed_mean.csv"
    )
    frame = pd.read_csv(path)
    frame = frame[
        frame["n_sources"].eq(1)
        & frame["sources"].isin(COMMON_GRAPHS)
        & frame["target"].isin(COMMON_GRAPHS)
    ].copy()
    grouped = frame[
        ["sources", "target", "roc_auc_mean", "roc_auc_sample_std", "training_seeds"]
    ].rename(
        columns={
            "sources": "source",
            "roc_auc_mean": "roc_auc",
            "roc_auc_sample_std": "sample_std",
            "training_seeds": "seeds",
        }
    )
    grouped["seeds"] = grouped["seeds"].map(
        lambda value: (
            int(value)
            if str(value).strip().isdigit()
            else len(str(value).split(","))
        )
    )
    grouped = grouped.assign(
        model="SAMGPT",
        objective="GraphCL",
        checkpoint_step=500,
        protocol="fixed downstream label-classification evaluation",
    )
    return _complete(grouped, "SAMGPT")


def load_graphsage(graphsage_root: Path) -> pd.DataFrame:
    path = graphsage_root / "results/matrix_s0/matrix_results.csv"
    frame = pd.read_csv(path)
    frame = frame[
        frame["checkpoint_step"].eq(2500)
        & frame["sources"].isin(COMMON_GRAPHS)
        & frame["target"].isin(COMMON_GRAPHS)
    ].copy()
    grouped = frame[
        ["sources", "target", "roc_auc_ovr_macro", "seed"]
    ].rename(
        columns={
            "sources": "source",
            "roc_auc_ovr_macro": "roc_auc",
        }
    )
    grouped["sample_std"] = np.nan
    grouped["seeds"] = 1
    grouped = grouped.drop(columns="seed").assign(
        model="GraphSAGE",
        objective="edge negative sampling",
        checkpoint_step=2500,
        protocol="fixed 10-labels/class linear probe",
    )
    return _complete(grouped, "GraphSAGE")


def _expanded_grid(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    metadata = frame.iloc[0][
        ["model", "objective", "checkpoint_step", "seeds", "protocol"]
    ].to_dict()
    expected = pd.MultiIndex.from_product(
        [SOCIAL_SOURCES, SOCIAL_TARGETS], names=["source", "target"]
    )
    if frame.duplicated(["source", "target"]).any():
        duplicates = frame.loc[
            frame.duplicated(["source", "target"], keep=False), ["source", "target"]
        ]
        raise ValueError(f"{model}: duplicate expanded cells: {duplicates.to_dict('records')}")
    expanded = frame.set_index(["source", "target"]).reindex(expected).reset_index()
    for column, value in metadata.items():
        expanded[column] = value
    expanded["available"] = expanded["roc_auc"].notna()
    return expanded


def load_prodigy_expanded() -> pd.DataFrame:
    path = (
        REPO
        / "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/"
        "data/prodigy_final_core/auc/summary/single_source_metrics_long.tsv"
    )
    frame = pd.read_csv(path, sep="\t")
    frame = frame[
        frame["source"].isin(SOCIAL_SOURCES)
        & frame["target"].isin(SOCIAL_TARGETS)
        & frame["checkpoint_step"].eq(2500)
    ]
    grouped = (
        frame.groupby(["source", "target"], as_index=False)
        .agg(
            roc_auc=("roc_auc_ovr_macro", "mean"),
            sample_std=("roc_auc_ovr_macro", "std"),
            seeds=("seed", "nunique"),
        )
        .assign(
            model="PRODIGY",
            objective="neighbor matching",
            checkpoint_step=2500,
            protocol="128 fixed 10-shot episodes",
        )
    )
    if len(grouped) != 45:
        raise ValueError(f"PRODIGY: expected 45 expanded cells, got {len(grouped)}")
    return _expanded_grid(grouped, "PRODIGY")


def load_vision_expanded() -> pd.DataFrame:
    specialist_path = (
        REPO
        / "scripts/experiments/analysis/transfer/matrices/cross_architecture/"
        "icl_arch_matrix/data/native_source_900_seed0/classification_all.tsv"
    )
    specialists = pd.read_csv(specialist_path, sep="\t")
    specialists = specialists[
        specialists["architecture"].eq("vision")
        & specialists["checkpoint_step"].eq(900)
        & specialists["dataset"].isin(SOCIAL_TARGETS)
    ].copy()
    specialists["source_list"] = specialists["sources"].map(ast.literal_eval)
    specialists = specialists[specialists["source_list"].map(len).eq(1)]
    specialists["source"] = specialists["source_list"].str[0]
    specialists = specialists[specialists["source"].isin(SOCIAL_SOURCES)]
    specialists = (
        specialists.rename(columns={"dataset": "target"})
        .groupby(["source", "target"], as_index=False)
        .agg(
            roc_auc=("roc_auc", "mean"),
            sample_std=("roc_auc", lambda _: np.nan),
            seeds=("training_seed", "nunique"),
        )
    )

    mixture_path = ROOT / "data/vision_native_mixture_cells.csv"
    mixture = pd.read_csv(mixture_path)
    mixture["source_list"] = mixture["sources"].map(ast.literal_eval)
    mixture = mixture[
        mixture["checkpoint_step"].eq(900)
        & mixture["source_list"].map(len).eq(1)
        & mixture["dataset"].isin(SOCIAL_TARGETS)
    ].copy()
    mixture["source"] = mixture["source_list"].str[0]
    # The mixture archive adds the otherwise absent UKR/RUS single-source row.
    mixture = mixture[mixture["source"].eq("ukr_rus")]
    mixture = (
        mixture.rename(columns={"dataset": "target"})
        .groupby(["source", "target"], as_index=False)
        .agg(
            roc_auc=("roc_auc", "mean"),
            sample_std=("roc_auc", lambda _: np.nan),
            seeds=("training_seed", "nunique"),
        )
    )

    completion_path = ROOT / "data/vision_social_source_completion.csv"
    completion = pd.read_csv(completion_path)
    completion = completion[
        completion["checkpoint_step"].eq(900)
        & completion["source"].isin({"covid", "cp_hk", "midterm"})
        & completion["target"].isin(SOCIAL_TARGETS)
    ].copy()
    fingerprints_per_target = completion.groupby("target")["episode_fingerprint"].nunique()
    if len(completion) != 15 or not fingerprints_per_target.eq(1).all():
        raise ValueError(
            "VISION social-source completion must contain 15 cells with one "
            "episode fingerprint per target"
        )
    completion = (
        completion.groupby(["source", "target"], as_index=False)
        .agg(
            roc_auc=("roc_auc", "mean"),
            sample_std=("roc_auc", lambda _: np.nan),
            seeds=("training_seed", "nunique"),
        )
    )

    grouped = pd.concat([specialists, mixture, completion], ignore_index=True).assign(
        model="VISION",
        objective="feature similarity",
        checkpoint_step=900,
        protocol="128 fixed 10-shot episodes",
    )
    if len(grouped) != 45:
        raise ValueError(f"VISION: expected 45 expanded cells, got {len(grouped)}")
    return _expanded_grid(grouped, "VISION")


def load_samgpt_expanded() -> pd.DataFrame:
    path = (
        REPO
        / "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/"
        "data/samgpt_downstream_cls/three_seed_mean.csv"
    )
    source_aliases = {
        "covid19_twitter": "covid",
        "cp_hk_twitter": "cp_hk",
        "ukr_rus_twitter": "ukr_rus",
    }
    target_aliases = {"facebook_page_category_top30": "facebook_page_reference"}
    raw_targets = set(SOCIAL_TARGETS) - {"facebook_page_reference"}
    raw_targets.add("facebook_page_category_top30")
    frame = pd.read_csv(path)
    frame = frame[frame["n_sources"].eq(1) & frame["target"].isin(raw_targets)].copy()
    frame["source"] = frame["sources"].replace(source_aliases)
    frame["target"] = frame["target"].replace(target_aliases)
    frame = frame[frame["source"].isin(SOCIAL_SOURCES)]
    grouped = frame[
        ["source", "target", "roc_auc_mean", "roc_auc_sample_std", "training_seeds"]
    ].rename(
        columns={
            "roc_auc_mean": "roc_auc",
            "roc_auc_sample_std": "sample_std",
            "training_seeds": "seeds",
        }
    )
    grouped["seeds"] = grouped["seeds"].map(
        lambda value: int(value) if str(value).strip().isdigit() else len(str(value).split(","))
    )
    grouped = grouped.assign(
        model="SAMGPT",
        objective="GraphCL",
        checkpoint_step=500,
        protocol="fixed downstream label-classification evaluation",
    )
    if len(grouped) != 45:
        raise ValueError(f"SAMGPT: expected 45 expanded cells, got {len(grouped)}")
    return _expanded_grid(grouped, "SAMGPT")


def load_graphsage_expanded(graphsage_root: Path) -> pd.DataFrame:
    path = graphsage_root / "results/matrix_s0/matrix_results.csv"
    frame = pd.read_csv(path)
    frame = frame[
        frame["checkpoint_step"].eq(2500)
        & frame["sources"].isin(SOCIAL_SOURCES)
        & frame["target"].isin(SOCIAL_TARGETS)
    ].copy()
    grouped = frame[
        ["sources", "target", "roc_auc_ovr_macro", "seed"]
    ].rename(
        columns={"sources": "source", "roc_auc_ovr_macro": "roc_auc"}
    )
    grouped["sample_std"] = np.nan
    grouped["seeds"] = 1
    grouped = grouped.drop(columns="seed").assign(
        model="GraphSAGE",
        objective="edge negative sampling",
        checkpoint_step=2500,
        protocol="fixed 10-labels/class linear probe",
    )
    if len(grouped) != 25:
        raise ValueError(f"GraphSAGE: expected 25 expanded cells, got {len(grouped)}")
    return _expanded_grid(grouped, "GraphSAGE")


def add_final_mixtures(frame: pd.DataFrame) -> pd.DataFrame:
    """Append the canonical final all-nine mixture for every plotted family."""
    path = ROOT / "data/native_specialists_vs_all9_cls.csv"
    mixture = pd.read_csv(path)
    mixture = mixture[
        mixture["kind"].eq("all-nine mixture")
        & mixture["target"].isin(SOCIAL_TARGETS)
    ].copy()
    mixture = mixture.rename(columns={"training_seeds": "seeds"})
    mixture["objective"] = mixture["model"].map(
        frame.groupby("model")["objective"].first()
    )
    mixture["protocol"] = mixture["model"].map(
        frame.groupby("model")["protocol"].first()
    )
    mixture["sample_std"] = np.nan
    mixture["available"] = True
    mixture = mixture[
        [
            "source", "target", "roc_auc", "sample_std", "model", "objective",
            "checkpoint_step", "seeds", "protocol", "available",
        ]
    ]
    expected = {"PRODIGY", "VISION", "SAMGPT"}
    observed = set(mixture["model"])
    counts = mixture.groupby("model")["target"].nunique().to_dict()
    if observed != expected or any(counts.get(model) != len(SOCIAL_TARGETS) for model in expected):
        raise ValueError(f"incomplete final-mixture rows: models={observed} counts={counts}")

    graphsage_path = ROOT / "data/graphsage_social_all9_cls.csv"
    graphsage = pd.read_csv(graphsage_path)
    expected_cells = {("all9", target) for target in SOCIAL_TARGETS}
    observed_cells = set(zip(graphsage["source"], graphsage["target"]))
    if observed_cells != expected_cells or len(graphsage) != len(expected_cells):
        raise ValueError(
            "GraphSAGE all-nine result must contain exactly the five social targets"
        )
    if not graphsage["checkpoint_step"].eq(2500).all():
        raise ValueError("GraphSAGE all-nine result is not the step-2,500 endpoint")
    if not graphsage["training_seed"].eq(0).all():
        raise ValueError("GraphSAGE all-nine result is not training seed 0")
    graphsage = graphsage.assign(
        sample_std=np.nan,
        seeds=1,
        model="GraphSAGE",
        objective="edge negative sampling",
        protocol="fixed 10-labels/class linear probe",
        available=True,
    )[
        [
            "source", "target", "roc_auc", "sample_std", "model", "objective",
            "checkpoint_step", "seeds", "protocol", "available",
        ]
    ]
    return pd.concat([frame, mixture, graphsage], ignore_index=True)


def plot(frame: pd.DataFrame, output: Path) -> None:
    models = ("PRODIGY", "VISION", "SAMGPT", "GraphSAGE")
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 10.4), constrained_layout=True)
    image = None
    for axis, model in zip(axes.flat, models):
        part = frame[frame["model"].eq(model)].copy()
        values = part.pivot(index="source", columns="target", values="roc_auc").reindex(
            index=COMMON_GRAPHS, columns=COMMON_GRAPHS
        )
        stds = part.pivot(index="source", columns="target", values="sample_std").reindex(
            index=COMMON_GRAPHS, columns=COMMON_GRAPHS
        )
        image = axis.imshow(values.to_numpy(), cmap="YlGnBu", vmin=0.45, vmax=1.0)
        for row in range(len(COMMON_GRAPHS)):
            for column in range(len(COMMON_GRAPHS)):
                value = values.iloc[row, column]
                std = stds.iloc[row, column]
                label = f"{value:.3f}"
                if not np.isnan(std):
                    label += f"\n±{std:.3f}"
                axis.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=9.2,
                    color="white" if value >= 0.77 else "#17212b",
                )
        for diagonal in range(len(COMMON_GRAPHS)):
            axis.add_patch(
                plt.Rectangle(
                    (diagonal - 0.49, diagonal - 0.49),
                    0.98,
                    0.98,
                    fill=False,
                    edgecolor="#e66101",
                    linewidth=2.1,
                )
            )
        metadata = part.iloc[0]
        axis.set_title(
            f"{model} · {metadata.objective}\n"
            f"step {int(metadata.checkpoint_step):,} · {int(metadata.seeds)} seed"
            f"{'s' if int(metadata.seeds) != 1 else ''}",
            fontsize=12.5,
        )
        axis.set_xticks(range(len(COMMON_GRAPHS)), [DISPLAY[name] for name in COMMON_GRAPHS])
        axis.set_yticks(range(len(COMMON_GRAPHS)), [DISPLAY[name] for name in COMMON_GRAPHS])
        axis.tick_params(axis="x", labelrotation=0, labelsize=9)
        axis.tick_params(axis="y", labelsize=9)
        axis.set_xlabel("Downstream label-classification target")
        axis.set_ylabel("Native-SSL source graph")
        axis.set_xticks(np.arange(-0.5, len(COMMON_GRAPHS), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(COMMON_GRAPHS), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.4)
        axis.tick_params(which="minor", bottom=False, left=False)

    if image is None:
        raise ValueError("no matrices plotted")
    colorbar = figure.colorbar(image, ax=axes, shrink=0.78, pad=0.025)
    colorbar.set_label("Downstream classification ROC-AUC")
    figure.suptitle(
        "Cross-graph transfer of native SSL specialists",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.015,
        "Orange outline: source graph equals target graph. Fixed-compute endpoints; "
        "probe protocols differ by family, so compare transfer structure rather than absolute rank.",
        ha="center",
        fontsize=9.5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_row(frame: pd.DataFrame, output: Path) -> None:
    models = ("PRODIGY", "VISION", "SAMGPT", "GraphSAGE")
    figure, axes = plt.subplots(
        1,
        len(models),
        figsize=(22.5, 5.8),
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for index, (axis, model) in enumerate(zip(axes, models)):
        part = frame[frame["model"].eq(model)].copy()
        values = part.pivot(index="source", columns="target", values="roc_auc").reindex(
            index=COMMON_GRAPHS, columns=COMMON_GRAPHS
        )
        stds = part.pivot(index="source", columns="target", values="sample_std").reindex(
            index=COMMON_GRAPHS, columns=COMMON_GRAPHS
        )
        image = axis.imshow(values.to_numpy(), cmap="YlGnBu", vmin=0.45, vmax=1.0)
        for row in range(len(COMMON_GRAPHS)):
            for column in range(len(COMMON_GRAPHS)):
                value = values.iloc[row, column]
                std = stds.iloc[row, column]
                label = f"{value:.3f}"
                if not np.isnan(std):
                    label += f"\n±{std:.3f}"
                axis.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8.7,
                    color="white" if value >= 0.77 else "#17212b",
                )
        for diagonal in range(len(COMMON_GRAPHS)):
            axis.add_patch(
                plt.Rectangle(
                    (diagonal - 0.49, diagonal - 0.49),
                    0.98,
                    0.98,
                    fill=False,
                    edgecolor="#e66101",
                    linewidth=2.0,
                )
            )
        metadata = part.iloc[0]
        panel = chr(ord("a") + index)
        axis.set_title(
            f"({panel}) {model} · {metadata.objective}\n"
            f"step {int(metadata.checkpoint_step):,} · {int(metadata.seeds)} seed"
            f"{'s' if int(metadata.seeds) != 1 else ''}",
            fontsize=11.8,
        )
        axis.set_xticks(range(len(COMMON_GRAPHS)), [DISPLAY[name] for name in COMMON_GRAPHS])
        axis.set_yticks(range(len(COMMON_GRAPHS)), [DISPLAY[name] for name in COMMON_GRAPHS])
        axis.tick_params(axis="x", labelrotation=0, labelsize=8.3)
        axis.tick_params(axis="y", labelsize=8.3, labelleft=index == 0)
        axis.set_xticks(np.arange(-0.5, len(COMMON_GRAPHS), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(COMMON_GRAPHS), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.3)
        axis.tick_params(which="minor", bottom=False, left=False)

    if image is None:
        raise ValueError("no matrices plotted")
    colorbar = figure.colorbar(image, ax=axes, shrink=0.72, pad=0.014)
    colorbar.set_label("Downstream classification ROC-AUC")
    figure.suptitle(
        "Cross-graph transfer of native SSL specialists",
        fontsize=16,
        fontweight="bold",
    )
    figure.supxlabel("Downstream label-classification target", fontsize=10.5)
    figure.supylabel("Native-SSL source graph", fontsize=10.5)
    figure.text(
        0.5,
        -0.025,
        "Orange outline: source graph equals target graph. Fixed-compute endpoints; "
        "probe protocols differ by family, so compare transfer structure rather than absolute rank.",
        ha="center",
        fontsize=9.2,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_expanded_row(frame: pd.DataFrame, output: Path) -> None:
    models = ("PRODIGY", "VISION", "SAMGPT", "GraphSAGE")
    colormap = plt.get_cmap("YlGnBu").copy()
    colormap.set_bad("#eeeeee")
    figure, axes = plt.subplots(
        1,
        len(models),
        figsize=(23.5, 10.0),
        sharey=True,
        constrained_layout=True,
    )
    image = None
    for index, (axis, model) in enumerate(zip(axes, models)):
        part = frame[frame["model"].eq(model)].copy()
        values = part.pivot(index="source", columns="target", values="roc_auc").reindex(
            index=PLOT_SOURCES, columns=SOCIAL_TARGETS
        )
        stds = part.pivot(index="source", columns="target", values="sample_std").reindex(
            index=PLOT_SOURCES, columns=SOCIAL_TARGETS
        )
        values["mean"] = values.mean(axis=1, skipna=True)
        values.loc[values[list(SOCIAL_TARGETS)].isna().all(axis=1), "mean"] = np.nan
        stds["mean"] = np.nan
        values = values.reindex(columns=PLOT_TARGETS)
        stds = stds.reindex(columns=PLOT_TARGETS)
        image = axis.imshow(
            np.ma.masked_invalid(values.to_numpy()),
            cmap=colormap,
            vmin=0.45,
            vmax=1.0,
            aspect="equal",
        )
        for row, source in enumerate(PLOT_SOURCES):
            for column, target in enumerate(PLOT_TARGETS):
                value = values.iloc[row, column]
                std = stds.iloc[row, column]
                if np.isnan(value):
                    axis.text(
                        column,
                        row,
                        "—",
                        ha="center",
                        va="center",
                        fontsize=9.5,
                        color="#777777",
                    )
                    continue
                label = f"{value:.3f}"
                if not np.isnan(std):
                    label += f"\n±{std:.3f}"
                axis.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7.6,
                    color="white" if value >= 0.77 else "#17212b",
                )
                if source == target:
                    axis.add_patch(
                        plt.Rectangle(
                            (column - 0.49, row - 0.49),
                            0.98,
                            0.98,
                            fill=False,
                            edgecolor="#e66101",
                            linewidth=2.0,
                        )
                    )
        axis.axvline(len(SOCIAL_TARGETS) - 0.5, color="#555555", linewidth=1.8)
        axis.axhline(len(SOCIAL_SOURCES) - 0.5, color="#555555", linewidth=1.8)
        metadata = part.iloc[0]
        available_sources = part.loc[
            part["available"] & part["source"].isin(SOCIAL_SOURCES), "source"
        ].nunique()
        panel = chr(ord("a") + index)
        axis.set_title(
            f"({panel}) {model} · {metadata.objective}\n"
            f"step {int(metadata.checkpoint_step):,} · {int(metadata.seeds)} seed"
            f"{'s' if int(metadata.seeds) != 1 else ''} · "
            f"{available_sources}/{len(SOCIAL_SOURCES)} source rows",
            fontsize=11.0,
        )
        axis.set_xticks(
            range(len(PLOT_TARGETS)), [DISPLAY[name] for name in PLOT_TARGETS]
        )
        axis.set_yticks(
            range(len(PLOT_SOURCES)), [DISPLAY[name] for name in PLOT_SOURCES]
        )
        axis.tick_params(axis="x", labelrotation=0, labelsize=7.7)
        axis.tick_params(axis="y", labelsize=8.0, labelleft=index == 0)
        axis.set_xticks(np.arange(-0.5, len(PLOT_TARGETS), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(PLOT_SOURCES), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.25)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.set_anchor("N")

    if image is None:
        raise ValueError("no expanded matrices plotted")
    colorbar = figure.colorbar(image, ax=axes, shrink=0.62, pad=0.012)
    colorbar.set_label("Downstream classification ROC-AUC")
    figure.suptitle(
        "Cross-graph transfer from native SSL specialists",
        fontsize=16,
        fontweight="bold",
    )
    figure.supxlabel("Downstream label-classification target", fontsize=10.5)
    figure.supylabel("Native-SSL source graph", fontsize=10.5)
    figure.text(
        0.5,
        -0.014,
        "Mean is unweighted across the five targets. Facebook uses page_category_top30 labels. "
        "* Source-only graph in this five-target label panel. Final mixture is all-nine; "
        "Orange outline: source=target; —: corresponding checkpoint unavailable. "
        "Fixed-compute endpoints; probe protocols differ by family.",
        ha="center",
        fontsize=8.8,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graphsage-root",
        type=Path,
        default=REPO.parent / "mixture-scaling",
        help="Standalone GraphSAGE mixture-scaling repository.",
    )
    args = parser.parse_args()

    frame = pd.concat(
        [load_prodigy(), load_vision(), load_samgpt(), load_graphsage(args.graphsage_root)],
        ignore_index=True,
    )
    data_path = ROOT / "data/cross_graph_cls_common_4x4.csv"
    frame.to_csv(data_path, index=False)
    plot(frame, ROOT / "figures/cross_graph_cls_common_4x4")
    plot_row(frame, ROOT / "figures/cross_graph_cls_common_4x4_row")

    expanded = pd.concat(
        [
            load_prodigy_expanded(),
            load_vision_expanded(),
            load_samgpt_expanded(),
            load_graphsage_expanded(args.graphsage_root),
        ],
        ignore_index=True,
    )
    expanded = add_final_mixtures(expanded)
    expanded.to_csv(ROOT / "data/cross_graph_cls_social_sources_9x5.csv", index=False)
    plot_expanded_row(
        expanded, ROOT / "figures/cross_graph_cls_social_sources_9x5_row"
    )
    print(
        "CROSS_GRAPH_CLS_OK "
        f"rows={len(frame)} models={frame.model.nunique()} "
        f"source_target_cells={frame.groupby('model').size().to_dict()} "
        f"expanded_rows={len(expanded)} "
        f"expanded_available={expanded.groupby('model').available.sum().to_dict()}"
    )


if __name__ == "__main__":
    main()
