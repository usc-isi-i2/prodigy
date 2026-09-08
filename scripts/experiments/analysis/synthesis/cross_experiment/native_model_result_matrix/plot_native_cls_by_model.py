#!/usr/bin/env python3
"""Plot native-pretraining checkpoint trajectories of downstream CLS by model."""

from __future__ import annotations

import csv
import ast
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[5]
FIGURES = ROOT / "figures"
COLORS = (
    "#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377",
    "#66CCEE", "#BBBBBB", "#000000", "#EE7733",
)
COMMON_TARGETS = (
    "covid_political", "election2020", "ukr_rus_suspended", "twibot20"
)
SPECIALIST_TARGETS = (*COMMON_TARGETS, "facebook_page_reference")
COMMON_TARGET_COLORS = dict(zip(SPECIALIST_TARGETS, COLORS))
LABELS = {
    "covid_political": "COVID political",
    "election2020": "Election 2020",
    "ukr_rus_suspended": "UKR/RUS suspended",
    "twibot20": "TwiBot-20",
    "facebook_page_reference": "Facebook pages",
    "facebook_page_verified": "Facebook verified",
    "cp_hk": "COVID protest HK",
    "covid": "COVID retweet",
    "midterm": "US midterm",
    "ukr_rus": "Ukraine/Russia",
    "cora": "Cora",
}


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def grouped_curves(
    rows: list[dict[str, str]], target_col: str, step_col: str, value_col: str
) -> dict[str, list[tuple[int, float]]]:
    curves: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        target = row[target_col]
        if target == "macro_target_mean":
            continue
        curves[target].append((int(row[step_col]), float(row[value_col])))
    return {target: sorted(points) for target, points in curves.items()}


def load_prodigy() -> dict[str, list[tuple[int, float]]]:
    path = (
        REPO
        / "scripts/experiments/analysis/transfer/matrices/cross_architecture/"
        "icl_arch_matrix/data/native_source_900_seed0/classification_summary.tsv"
    )
    rows = [row for row in read_csv(path, "\t") if row["architecture"] == "prodigy"]
    return grouped_curves(rows, "target", "checkpoint_step", "mean_roc_auc")


def load_vision() -> dict[str, list[tuple[int, float]]]:
    rows = read_csv(ROOT / "data/vision_all9_saturation_summary.csv")
    return grouped_curves(rows, "dataset", "checkpoint_step", "roc_auc_mean")


def load_samgpt() -> dict[str, list[tuple[int, float]]]:
    rows = read_csv(ROOT / "data/samgpt_all9_saturation_summary.csv")
    return grouped_curves(rows, "target", "checkpoint_update", "roc_auc_mean")


def load_graphsage() -> dict[str, list[tuple[int, float]]]:
    rows = read_csv(ROOT / "data/graphsage_matched_saturation_endpoint_by_target.csv")
    return grouped_curves(rows, "target", "pretraining_updates", "roc_auc_mean")


def plot_model(model: str, objective: str, curves: dict[str, list[tuple[int, float]]]) -> None:
    steps = sorted({step for points in curves.values() for step, _ in points})
    positions = {step: index for index, step in enumerate(steps)}
    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    for (target, points), color in zip(curves.items(), COLORS):
        axis.plot(
            [positions[step] for step, _ in points],
            [value for _, value in points],
            color=color,
            marker="o",
            markersize=5,
            linewidth=2.2,
            label=LABELS.get(target, target.replace("_", " ").title()),
        )
    axis.set_xticks(range(len(steps)), [str(step) for step in steps])
    axis.set_xlabel("Native-pretraining updates")
    axis.set_ylabel("Downstream classification ROC-AUC")
    axis.set_ylim(0.25, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2, fontsize=9)
    axis.set_title(f"{model}: {objective} → downstream classification")
    fig.tight_layout()
    stem = FIGURES / f"{model.lower()}_native_cls_by_target_colored"
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_combined(
    models: list[tuple[str, str, dict[str, list[tuple[int, float]]]]]
) -> None:
    common_targets = [
        target
        for target in ("covid_political", "election2020", "ukr_rus_suspended", "twibot20")
        if all(target in curves for _, _, curves in models)
    ]
    fig, axes = plt.subplots(1, len(models), figsize=(24, 5.4), sharey=True)
    for index, (axis, (model, objective, curves)) in enumerate(zip(axes, models)):
        steps = sorted({step for target in common_targets for step, _ in curves[target]})
        positions = {step: position for position, step in enumerate(steps)}
        for target, color in zip(common_targets, COLORS):
            points = curves[target]
            axis.plot(
                [positions[step] for step, _ in points],
                [value for _, value in points],
                color=color,
                marker="o",
                markersize=4,
                linewidth=2,
                label=LABELS.get(target, target.replace("_", " ").title()),
            )
        axis.set_xticks(range(len(steps)), [str(step) for step in steps])
        axis.set_xlabel("Pretraining updates")
        axis.set_ylim(0.25, 1.0)
        axis.grid(axis="y", alpha=0.25)
        axis.set_title(f"{model}\n{objective}")
        axis.legend(frameon=False, fontsize=7.3, ncol=1, loc="best")
        if index == 0:
            axis.set_ylabel("Downstream classification ROC-AUC")
    fig.suptitle("Native pretraining → downstream classification", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    stem = FIGURES / "native_cls_by_model_side_by_side"
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_common_targets(
    models: list[tuple[str, str, dict[str, list[tuple[int, float]]]]]
) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(24, 5.4), sharey=True)
    for index, (axis, (model, objective, curves)) in enumerate(zip(axes, models)):
        common_curves = {target: curves[target] for target in COMMON_TARGETS}
        steps = sorted({step for points in common_curves.values() for step, _ in points})
        positions = {step: position for position, step in enumerate(steps)}
        for target, points in common_curves.items():
            axis.plot(
                [positions[step] for step, _ in points],
                [value for _, value in points],
                color=COMMON_TARGET_COLORS[target],
                marker="o",
                markersize=4,
                linewidth=2,
                label=LABELS[target],
            )
        axis.set_xticks(range(len(steps)), [str(step) for step in steps])
        axis.set_xlabel("Pretraining updates")
        axis.set_ylim(0.25, 1.0)
        axis.grid(axis="y", alpha=0.25)
        axis.set_title(f"{model}\n{objective}")
        axis.legend(frameon=False, fontsize=8, loc="best")
        if index == 0:
            axis.set_ylabel("Downstream classification ROC-AUC")
    fig.suptitle("Native pretraining → downstream classification: common targets", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    stem = FIGURES / "native_cls_by_model_common_targets_side_by_side"
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_target_specialists(include_ssl: bool = False) -> None:
    path = (
        REPO
        / "scripts/experiments/analysis/transfer/matrices/cross_architecture/"
        "icl_arch_matrix/data/native_source_900_seed0/classification_all.tsv"
    )
    rows = read_csv(path, "\t")
    prodigy_seed2_dir = (
        REPO
        / "scripts/experiments/analysis/transfer/matrices/cross_architecture/"
        "icl_arch_matrix/data/finalcore_cls2500_seed2"
    )
    prodigy_seed2_rows = read_csv(
        prodigy_seed2_dir / "classification_self_trajectory.tsv", "\t"
    ) + read_csv(prodigy_seed2_dir / "classification_auc.tsv", "\t")
    if include_ssl:
        fig, grid = plt.subplots(2, 4, figsize=(24, 10.2))
        axes = grid[0]
    else:
        fig, axes = plt.subplots(1, 4, figsize=(24, 5.4), sharey=True)
    available = (
        ("PRODIGY", "prodigy", "neighbor matching (seed 2; init seed 0)"),
        ("VISION", "vision", "feature similarity"),
    )
    for axis, (model, architecture, objective) in zip(axes[:2], available):
        for target in SPECIALIST_TARGETS:
            if architecture == "prodigy":
                points = [
                    (int(row["checkpoint_step"]), float(row["roc_auc"]))
                    for row in prodigy_seed2_rows
                    if row["model_id"] == f"ss_{target}" and row["dataset"] == target
                ]
                points.extend(
                    (0, float(row["roc_auc"]))
                    for row in rows
                    if row["architecture"] == "prodigy"
                    and row["dataset"] == target
                    and row["baseline"] == "random_init"
                    and int(row["checkpoint_step"]) == 0
                )
                if sorted(step for step, _ in points) != [0, 300, 900, 2500]:
                    raise ValueError(f"PRODIGY {target}: incomplete seed-2 trajectory")
            else:
                points = []
                for row in rows:
                    if row["architecture"] != architecture or row["dataset"] != target:
                        continue
                    step = int(row["checkpoint_step"])
                    if step == 0 and row["baseline"] == "random_init":
                        points.append((step, float(row["roc_auc"])))
                    elif (
                        step in {100, 300, 900}
                        and row["baseline"] != "random_init"
                        and ast.literal_eval(row["sources"]) == [target]
                    ):
                        points.append((step, float(row["roc_auc"])))
                if sorted(step for step, _ in points) != [0, 100, 300, 900]:
                    raise ValueError(f"VISION {target}: incomplete displayed trajectory")
            points.sort()
            axis.plot(
                range(len(points)), [value for _, value in points],
                color=COMMON_TARGET_COLORS[target], marker="o", linewidth=2.2,
                label=LABELS[target],
            )
        steps = [step for step, _ in points]
        axis.set_xticks(range(len(steps)), [str(step) for step in steps])
        axis.set_xlabel("Pretraining updates")
        axis.set_ylim(0.25, 1.0)
        axis.grid(axis="y", alpha=0.25)
        axis.set_title(f"{model}\n{objective}")
        axis.legend(frameon=False, fontsize=8, loc="best")
    axes[0].set_ylabel("Downstream classification ROC-AUC")
    samgpt_rows = read_csv(ROOT / "data/samgpt_specialist_cls_trajectory.csv")
    samgpt_axis = axes[2]
    samgpt_steps = (0, 60, 180, 500)
    for target in SPECIALIST_TARGETS:
        target_rows = sorted(
            (
                row for row in samgpt_rows
                if row["target"] == target
                and int(row["checkpoint_update"]) in samgpt_steps
            ),
            key=lambda row: int(row["checkpoint_update"]),
        )
        if [int(row["checkpoint_update"]) for row in target_rows] != list(samgpt_steps):
            raise ValueError(f"SAMGPT {target}: incomplete specialist trajectory")
        if any(int(row["training_seeds"]) != 3 for row in target_rows):
            raise ValueError(f"SAMGPT {target}: expected three training seeds")
        samgpt_axis.plot(
            range(len(samgpt_steps)),
            [float(row["roc_auc_mean"]) for row in target_rows],
            color=COMMON_TARGET_COLORS[target], marker="o", linewidth=2.2,
            label=LABELS[target],
        )
    samgpt_axis.set_xticks(range(len(samgpt_steps)), [str(step) for step in samgpt_steps])
    samgpt_axis.set_xlabel("Pretraining updates")
    samgpt_axis.set_ylim(0.25, 1.0)
    samgpt_axis.grid(axis="y", alpha=0.25)
    samgpt_axis.set_title("SAMGPT\nGraphCL (3-seed mean)")
    samgpt_axis.legend(frameon=False, fontsize=8, loc="best")

    graphsage_rows = []
    for graph_path in (
        Path("/Users/philipp/projects/gfm/mixture-scaling/results/primary_s0/primary_results.csv"),
        Path("/Users/philipp/projects/gfm/mixture-scaling/results/primary_s1_s2/primary_results.csv"),
    ):
        graphsage_rows.extend(read_csv(graph_path))
    graphsage_axis = axes[3]
    graphsage_steps = (300, 900, 2500)
    graphsage_step0 = {
        row["target"]: float(row["roc_auc_mean"])
        for row in read_csv(ROOT / "data/graphsage_specialist_step0_cls.csv")
    }
    for target in SPECIALIST_TARGETS:
        values = []
        for step in graphsage_steps:
            seed_values = [
                float(row["roc_auc_ovr_macro"])
                for row in graphsage_rows
                if row["target"] == target
                and row["sources"] == target
                and int(row["checkpoint_step"]) == step
            ]
            if len(seed_values) != 3:
                raise ValueError(f"GraphSAGE {target}/{step}: expected three seeds")
            values.append(statistics.mean(seed_values))
        display_steps = (0, *graphsage_steps)
        values = [graphsage_step0[target], *values]
        graphsage_axis.plot(
            range(len(display_steps)), values,
            color=COMMON_TARGET_COLORS[target], marker="o", linewidth=2.2,
            label=LABELS[target],
        )
    graphsage_axis.set_xticks(range(len(display_steps)), [str(step) for step in display_steps])
    graphsage_axis.set_xlabel("Pretraining updates")
    graphsage_axis.set_ylim(0.25, 1.0)
    graphsage_axis.grid(axis="y", alpha=0.25)
    graphsage_axis.set_title("GraphSAGE\nlink prediction (3-seed mean)")
    graphsage_axis.legend(frameon=False, fontsize=8, loc="best")
    if include_ssl:
        loss_rows = read_csv(ROOT / "data/specialist_training_loss_provisional.csv")
        model_steps = {
            "PRODIGY": (0, 300, 900, 2500),
            "VISION": (0, 100, 300, 900),
            "SAMGPT": (0, 60, 180, 500),
            "GraphSAGE": (0, 300, 900, 2500),
        }
        objectives = {
            "PRODIGY": "neighbor-matching loss",
            "VISION": "feature-similarity loss",
            "SAMGPT": "GraphCL loss",
        }
        for index, model in enumerate(("PRODIGY", "VISION", "SAMGPT", "GraphSAGE")):
            axis = grid[1, index]
            steps = model_steps[model]
            for target in SPECIALIST_TARGETS:
                selected = {
                    int(row["step"]): float(row["loss"])
                    for row in loss_rows
                    if row["model"] == model and row["target"] == target
                }
                values = [selected[step] for step in steps]
                axis.plot(
                    range(len(steps)), values, color=COMMON_TARGET_COLORS[target],
                    marker="o", linewidth=2.2, label=LABELS[target],
                )
            axis.set_xticks(range(len(steps)), [str(step) for step in steps])
            axis.set_xlabel("Pretraining updates")
            axis.set_title(objectives.get(model, "link-prediction training loss"))
            axis.grid(axis="y", alpha=0.25)
        grid[1, 0].set_ylabel("Native SSL loss")
        fig.suptitle("Target-specialist classification and native SSL loss", fontsize=17)
        stem = FIGURES / "native_cls_ssl_target_specialists_side_by_side"
    else:
        fig.suptitle(
            "Target-specialist native pretraining → classification on the same target",
            fontsize=17,
        )
        stem = FIGURES / "native_cls_target_specialists_side_by_side"
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_mean_cls_and_ssl() -> None:
    classification_path = (
        REPO
        / "scripts/experiments/analysis/transfer/matrices/cross_architecture/"
        "icl_arch_matrix/data/native_source_900_seed0/classification_all.tsv"
    )
    classification_rows = read_csv(classification_path, "\t")
    seed2_dir = (
        REPO
        / "scripts/experiments/analysis/transfer/matrices/cross_architecture/"
        "icl_arch_matrix/data/finalcore_cls2500_seed2"
    )
    seed2_rows = read_csv(seed2_dir / "classification_self_trajectory.tsv", "\t")
    seed2_rows += read_csv(seed2_dir / "classification_auc.tsv", "\t")

    auc: dict[str, dict[int, float]] = {}
    prodigy_values: dict[int, list[float]] = defaultdict(list)
    for row in classification_rows:
        if (
            row["architecture"] == "prodigy"
            and row["baseline"] == "random_init"
            and row["dataset"] in SPECIALIST_TARGETS
        ):
            prodigy_values[0].append(float(row["roc_auc"]))
    for row in seed2_rows:
        if row["dataset"] in SPECIALIST_TARGETS and row["model_id"] == f"ss_{row['dataset']}":
            prodigy_values[int(row["checkpoint_step"])].append(float(row["roc_auc"]))
    auc["PRODIGY"] = {step: statistics.mean(values) for step, values in prodigy_values.items()}

    vision_values: dict[int, list[float]] = defaultdict(list)
    for row in classification_rows:
        if row["architecture"] != "vision" or row["dataset"] not in SPECIALIST_TARGETS:
            continue
        step = int(row["checkpoint_step"])
        if step == 0 and row["baseline"] == "random_init":
            vision_values[step].append(float(row["roc_auc"]))
        elif step in {100, 300, 900} and ast.literal_eval(row["sources"]) == [row["dataset"]]:
            vision_values[step].append(float(row["roc_auc"]))
    auc["VISION"] = {step: statistics.mean(values) for step, values in vision_values.items()}

    samgpt_values: dict[int, list[float]] = defaultdict(list)
    for row in read_csv(ROOT / "data/samgpt_specialist_cls_trajectory.csv"):
        if row["target"] in SPECIALIST_TARGETS and int(row["checkpoint_update"]) in {0, 60, 180, 500}:
            samgpt_values[int(row["checkpoint_update"])].append(float(row["roc_auc_mean"]))
    auc["SAMGPT"] = {step: statistics.mean(values) for step, values in samgpt_values.items()}

    loss_values: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in read_csv(ROOT / "data/specialist_training_loss_provisional.csv"):
        loss_values[row["model"]][int(row["step"])].append(float(row["loss"]))
    losses = {
        model: {step: statistics.mean(values) for step, values in by_step.items()}
        for model, by_step in loss_values.items()
    }

    steps_by_model = {
        "PRODIGY": (0, 300, 900, 2500),
        "VISION": (0, 100, 300, 900),
        "SAMGPT": (0, 60, 180, 500),
    }
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.4), sharey=True)
    for index, (axis, model) in enumerate(zip(axes, steps_by_model)):
        steps = steps_by_model[model]
        positions = range(len(steps))
        auc_line = axis.plot(
            positions, [auc[model][step] for step in steps], color="#4477AA",
            linewidth=3.2, label="Mean ROC-AUC",
        )[0]
        loss_axis = axis.twinx()
        loss_line = loss_axis.plot(
            positions, [losses[model][step] for step in steps], color="#CC6677",
            linewidth=3.2, label="Mean SSL loss",
        )[0]
        axis.set_xticks(list(positions), [str(step) for step in steps], fontsize=14)
        axis.tick_params(axis="y", labelsize=14)
        axis.set_xlabel("Pretraining updates", fontsize=16)
        axis.set_ylim(0.35, 1.0)
        axis.grid(axis="y", alpha=0.25)
        axis.set_title(model, fontsize=19)
        loss_axis.set_ylabel("Mean native SSL loss", color="#CC6677", fontsize=16)
        loss_axis.tick_params(axis="y", colors="#CC6677", labelsize=14)
        axis.legend(
            [auc_line, loss_line], ["Mean ROC-AUC", "Mean SSL loss"],
            frameon=False, fontsize=14,
        )
        if index == 0:
            axis.set_ylabel(
                "Mean downstream classification ROC-AUC", color="#4477AA", fontsize=16,
            )
    fig.suptitle("Mean target-specialist classification and native SSL loss", fontsize=22)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    stem = FIGURES / "native_cls_ssl_target_specialists_means_side_by_side"
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    models = [
        ("PRODIGY", "neighbor matching", load_prodigy()),
        ("VISION", "feature similarity", load_vision()),
        ("SAMGPT", "GraphCL", load_samgpt()),
        ("GraphSAGE", "link prediction", load_graphsage()),
    ]
    for model, objective, curves in models:
        plot_model(model, objective, curves)
    plot_combined(models)
    plot_common_targets(models)
    plot_target_specialists()
    plot_target_specialists(include_ssl=True)
    plot_mean_cls_and_ssl()


if __name__ == "__main__":
    main()
