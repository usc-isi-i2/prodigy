#!/usr/bin/env python3
"""Compare composition, mixture-size, graph-choice, and compute explanations."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import plot_max_rule_downstream as source


HERE = Path(__file__).resolve().parent
OUT_DATA = HERE / "data/mixture_explanations"
OUT_FIGURES = HERE / "figures/pngs"
DOWNSTREAM = (
    source.REPO
    / "scripts/experiments/analysis/transfer/ablations/prodigy_nm/downstream/"
    "nm_ladder_downstream_nhop2"
)
MATCHED_NM = (
    source.REPO
    / "scripts/experiments/analysis/transfer/ablations/prodigy_nm/source_exposure/"
    "nm_ladder_fixed_exposure_nhop2/data/comparison_to_matched40k_h2_orderA.csv"
)
SATURATION = (
    source.REPO
    / "scripts/experiments/analysis/transfer/ablations/prodigy_nm/saturation/"
    "pretrain_saturation_nhop2/data/nhop_comparison.csv"
)

SETTING_LABELS = {
    "fixed_compute_nm": "Fixed compute · NM",
    "fixed_compute_classification": "Fixed compute · classification",
    "fixed_exposure_nm": "Fixed exposure · NM",
    "fixed_exposure_classification": "Fixed exposure · classification",
}
SETTING_COLORS = {
    "fixed_compute_nm": "#0072B2",
    "fixed_compute_classification": "#56B4E9",
    "fixed_exposure_nm": "#D55E00",
    "fixed_exposure_classification": "#E69F00",
}
MODEL_SPECS = {
    "max": (),
    "second_best": (),
    "mean": (),
    "softmax": (),
    "max_plus_size": ("best", "log_n_sources"),
    "max_plus_rest": ("best", "mean_rest"),
    "max_plus_diversity": ("best", "profile_diversity"),
    "max_plus_size_rest": ("best", "log_n_sources", "mean_rest"),
    "max_plus_size_rest_diversity": ("best", "log_n_sources", "mean_rest", "profile_diversity"),
    "max_plus_size_rest_spread": ("best", "log_n_sources", "mean_rest", "score_std"),
    "composition_full": ("best", "log_n_sources", "mean_rest", "score_std", "profile_diversity"),
}


def _specialist_lookup(task: str, fixed_compute: bool) -> dict[tuple[str, str], float]:
    if task == "nm" and fixed_compute:
        data = pd.read_csv(source.FIXED_COMPUTE_NM_SS, sep="\t")
        data = data[data.target.isin(source.TARGETS)]
        return data.groupby(["source", "target"]).roc_auc_ovr_macro.mean().to_dict()
    if task == "nm":
        return source.historical_nm_specialists()
    return source.historical_classification_specialists()


def _profile_diversity(sources: tuple[str, ...], lookup: dict[tuple[str, str], float]) -> float:
    profiles = {
        graph: np.array([lookup[(graph, target)] for target in source.TARGETS], dtype=float)
        for graph in sources
    }
    distances = [
        float(np.sqrt(np.mean((profiles[left] - profiles[right]) ** 2)))
        for left, right in itertools.combinations(sources, 2)
    ]
    return float(np.mean(distances)) if distances else 0.0


def _feature_row(
    setting: str,
    task: str,
    order: str,
    rung: int,
    target: str,
    sources: tuple[str, ...],
    observed: float,
    total_updates: int,
    lookup: dict[tuple[str, str], float],
) -> dict[str, object]:
    scores = np.sort(np.array([lookup[(graph, target)] for graph in sources], dtype=float))[::-1]
    if len(scores) < 2:
        raise ValueError("mixture analysis requires at least two constituents")
    rest = scores[1:]
    return {
        "setting": setting,
        "task": task,
        "order": order,
        "rung": int(rung),
        "target": target,
        "sources": ",".join(sources),
        "n_sources": len(sources),
        "total_updates": int(total_updates),
        "log_n_sources": float(np.log(len(sources))),
        "log_total_updates": float(np.log(total_updates)),
        "target_in_sources": int(target in sources),
        "observed_auc": float(observed),
        "constituent_scores": ";".join(f"{value:.12g}" for value in scores),
        "best": float(scores[0]),
        "second_best": float(scores[1]),
        "mean": float(scores.mean()),
        "median": float(np.median(scores)),
        "mean_rest": float(rest.mean()),
        "score_std": float(scores.std(ddof=0)),
        "score_range": float(scores.max() - scores.min()),
        "profile_diversity": _profile_diversity(sources, lookup),
        "trajectory": f"{target}|{order}",
    }


def fixed_compute_rows(task: str) -> list[dict[str, object]]:
    setting = f"fixed_compute_{task}"
    lookup = _specialist_lookup(task, fixed_compute=True)
    plan = source.read_plan()
    if task == "nm":
        values = pd.read_csv(source.FIXED_COMPUTE_NM, sep="\t")
        values = values[values.target.isin(source.TARGETS)].groupby(
            ["model_id", "target"], as_index=False
        ).roc_auc_ovr_macro_logged.mean()
        target_column, value_column = "target", "roc_auc_ovr_macro_logged"
    else:
        values = pd.read_csv(source.FIXED_COMPUTE_CLS, sep="\t")
        values = values[values.dataset.isin(source.TARGETS)].groupby(
            ["model_id", "dataset"], as_index=False
        ).roc_auc.mean()
        target_column, value_column = "dataset", "roc_auc"
    merged = plan.merge(values, on="model_id", validate="many_to_many")
    rows = []
    for raw in merged.itertuples(index=False):
        if raw.rung < 2:
            continue
        target = getattr(raw, target_column)
        rows.append(
            _feature_row(
                setting, task, raw.order, raw.rung, target, tuple(raw.sources),
                getattr(raw, value_column), 2_500, lookup,
            )
        )
    return rows


def fixed_exposure_rows(task: str) -> list[dict[str, object]]:
    setting = f"fixed_exposure_{task}"
    lookup = _specialist_lookup(task, fixed_compute=False)
    if task == "nm":
        data = pd.read_csv(source.FIXED_EXPOSURE_NM)
        data = data[data.dataset.isin(source.TARGETS)].copy()
        data["target"] = data.dataset
        data["observed"] = data.test_roc_auc
    else:
        wide = pd.read_csv(source.FIXED_EXPOSURE_CLS)
        data = wide[wide.variant.eq("fixed10k")].melt(
            id_vars=["order", "rung", "sources"], value_vars=list(source.TARGETS),
            var_name="target", value_name="observed",
        )
    rows = []
    for raw in data.itertuples(index=False):
        if raw.rung < 2:
            continue
        sources = tuple(source.normalize_source(graph) for graph in raw.sources.split(","))
        rows.append(
            _feature_row(
                setting, task, raw.order, raw.rung, raw.target, sources,
                raw.observed, 10_000 * raw.rung, lookup,
            )
        )
    return rows


def build_canonical() -> pd.DataFrame:
    rows = []
    for task in ("nm", "classification"):
        rows.extend(fixed_compute_rows(task))
        rows.extend(fixed_exposure_rows(task))
    frame = pd.DataFrame(rows).sort_values(["setting", "target", "order", "rung"])
    expected = {
        "fixed_compute_nm": 96,
        "fixed_compute_classification": 96,
        "fixed_exposure_nm": 56,
        "fixed_exposure_classification": 56,
    }
    if frame.groupby("setting").size().to_dict() != expected:
        raise ValueError(f"unexpected canonical coverage: {frame.groupby('setting').size().to_dict()}")
    if frame.duplicated(["setting", "order", "rung", "target"]).any():
        raise ValueError("duplicate canonical mixture cells")
    return frame


def _fit_linear(train: pd.DataFrame, test: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    usable = [column for column in columns if train[column].std(ddof=0) > 1e-12]
    x_train = np.column_stack([np.ones(len(train))] + [train[column].to_numpy(float) for column in usable])
    x_test = np.column_stack([np.ones(len(test))] + [test[column].to_numpy(float) for column in usable])
    coefficient = np.linalg.lstsq(x_train, train.observed_auc.to_numpy(float), rcond=None)[0]
    return x_test @ coefficient


def _softmax_prediction(frame: pd.DataFrame, beta: float) -> np.ndarray:
    predictions = []
    for row in frame.itertuples(index=False):
        values = np.fromstring(row.constituent_scores, sep=";")
        weights = np.exp(beta * (values - values.max()))
        predictions.append(float(np.sum(values * weights) / np.sum(weights)))
    return np.array(predictions)


def grouped_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for setting, data in frame.groupby("setting"):
        for held_out_target in source.TARGETS:
            train = data[data.target.ne(held_out_target)]
            test = data[data.target.eq(held_out_target)]
            if test.empty:
                continue
            direct = {
                "max": test.best.to_numpy(float),
                "second_best": test.second_best.to_numpy(float),
                "mean": test["mean"].to_numpy(float),
            }
            beta_grid = np.concatenate(([0.0], np.geomspace(.01, 300, 240)))
            train_y = train.observed_auc.to_numpy(float)
            beta = min(
                beta_grid,
                key=lambda candidate: np.mean(np.abs(_softmax_prediction(train, candidate) - train_y)),
            )
            direct["softmax"] = _softmax_prediction(test, float(beta))
            for model, columns in MODEL_SPECS.items():
                prediction = direct[model] if model in direct else _fit_linear(train, test, columns)
                for row_index, value in zip(test.index, prediction):
                    outputs.append(
                        {
                            "row_index": row_index,
                            "setting": setting,
                            "held_out_target": held_out_target,
                            "model": model,
                            "prediction": float(value),
                            "softmax_beta": float(beta) if model == "softmax" else np.nan,
                        }
                    )
    predictions = pd.DataFrame(outputs)
    observed = frame[["observed_auc", "trajectory", "target", "order", "rung"]]
    predictions = predictions.join(observed, on="row_index")
    predictions["error"] = predictions.prediction - predictions.observed_auc
    predictions["absolute_error"] = predictions.error.abs()
    predictions["squared_error"] = predictions.error**2
    return predictions


def summarize_cv(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(["setting", "model"], as_index=False)
        .agg(
            mae=("absolute_error", "mean"),
            rmse=("squared_error", lambda values: float(np.sqrt(values.mean()))),
            bias=("error", "mean"),
            pearson=("prediction", lambda values: np.nan),
            cells=("prediction", "size"),
        )
        .drop(columns="pearson")
        .merge(
            predictions.groupby(["setting", "model"]).apply(
                lambda part: pearsonr(part.prediction, part.observed_auc).statistic,
                include_groups=False,
            ).rename("pearson").reset_index(),
            on=["setting", "model"],
        )
    )


def exact_cluster_signflip(cluster_differences: np.ndarray) -> float:
    cluster_differences = np.asarray(cluster_differences, dtype=float)
    observed = abs(cluster_differences.mean())
    if len(cluster_differences) <= 16:
        values = []
        for signs in itertools.product((-1.0, 1.0), repeat=len(cluster_differences)):
            values.append(abs(np.mean(cluster_differences * np.asarray(signs))))
        return float(np.mean(np.asarray(values) >= observed - 1e-15))
    rng = np.random.default_rng(20260819)
    signs = rng.choice((-1.0, 1.0), size=(100_000, len(cluster_differences)))
    return float(np.mean(np.abs((signs * cluster_differences).mean(axis=1)) >= observed))


def bootstrap_cluster_ci(cluster_differences: pd.Series, draws: int = 20_000) -> tuple[float, float]:
    values = cluster_differences.to_numpy(float)
    rng = np.random.default_rng(20260819)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(sampled, [.025, .975]))


def paired_model_tests(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    max_loss = predictions[predictions.model.eq("max")][["row_index", "absolute_error", "trajectory"]].rename(
        columns={"absolute_error": "max_loss"}
    )
    for (setting, model), part in predictions[predictions.model.ne("max")].groupby(["setting", "model"]):
        merged = part.merge(max_loss, on=["row_index", "trajectory"], validate="one_to_one")
        merged["loss_difference"] = merged.absolute_error - merged.max_loss
        cluster = merged.groupby("trajectory").loss_difference.mean()
        low, high = bootstrap_cluster_ci(cluster)
        rows.append(
            {
                "setting": setting,
                "model": model,
                "delta_mae_vs_max": merged.loss_difference.mean(),
                "cluster_bootstrap_low": low,
                "cluster_bootstrap_high": high,
                "cluster_signflip_p_two_sided": exact_cluster_signflip(cluster.to_numpy()),
                "trajectory_clusters": len(cluster),
            }
        )
    return pd.DataFrame(rows)


def size_and_choice_tests(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    size_rows = []
    choice_rows = []
    rng = np.random.default_rng(20260819)
    for setting, data in frame.groupby("setting"):
        residual = data.observed_auc - data.best
        rho, p = spearmanr(data.n_sources, residual)
        trajectory_slopes = []
        for _, part in data.groupby("trajectory"):
            trajectory_slopes.append(np.polyfit(part.log_n_sources, part.observed_auc - part.best, 1)[0])
        slopes = pd.Series(trajectory_slopes)
        low, high = bootstrap_cluster_ci(slopes)
        size_rows.append(
            {
                "setting": setting,
                "spearman_n_vs_max_residual": rho,
                "naive_spearman_p": p,
                "mean_trajectory_slope_per_log_n": slopes.mean(),
                "cluster_bootstrap_low": low,
                "cluster_bootstrap_high": high,
                "trajectory_clusters": len(slopes),
            }
        )

        centered = data.copy()
        for column in ("observed_auc", "best", "profile_diversity"):
            centered[f"{column}_centered"] = centered[column] - centered.groupby(
                ["target", "n_sources"]
            )[column].transform("mean")
        valid = centered.best_centered.abs().gt(1e-12) | centered.observed_auc_centered.abs().gt(1e-12)
        centered = centered[valid]
        observed_corr = pearsonr(centered.best_centered, centered.observed_auc_centered).statistic
        x_values = centered.best_centered.to_numpy(float)
        y_values = centered.observed_auc_centered.to_numpy(float)
        group_indices = [
            np.asarray(indices, dtype=int)
            for indices in centered.reset_index(drop=True).groupby(["target", "n_sources"]).indices.values()
        ]
        permutation_values = []
        for _ in range(20_000):
            permuted = x_values.copy()
            for indices in group_indices:
                permuted[indices] = rng.permutation(permuted[indices])
            permutation_values.append(np.corrcoef(permuted, y_values)[0, 1])
        permutation_values = np.asarray(permutation_values)
        permutation_p = float((1 + np.sum(np.abs(permutation_values) >= abs(observed_corr))) / (len(permutation_values) + 1))
        choice_rows.append(
            {
                "setting": setting,
                "within_target_size_best_choice_correlation": observed_corr,
                "stratified_permutation_p_two_sided": permutation_p,
                "cells": len(centered),
                "orders": centered.order.nunique(),
            }
        )
    return pd.DataFrame(size_rows), pd.DataFrame(choice_rows)


def schedule_comparison() -> tuple[pd.DataFrame, pd.DataFrame]:
    nm = pd.read_csv(MATCHED_NM)
    nm_rows = nm.assign(
        task="nm",
        target=nm.dataset.map(source.normalize_source),
        fixed_exposure_auc=nm.fixed_exposure_h2_auc,
        matched40k_auc=nm.matched40k_h2_auc,
    )[["task", "rung", "target", "fixed_total_steps", "matched40k_total_steps", "fixed_exposure_auc", "matched40k_auc"]]

    classification = pd.read_csv(source.FIXED_EXPOSURE_CLS)
    fixed = classification[classification.variant.eq("fixed10k") & classification.order.eq("A")]
    matched = classification[classification.variant.eq("matched40k") & classification.order.eq("A")]
    id_columns = ["rung"]
    fixed_long = fixed.melt(id_vars=id_columns, value_vars=list(source.TARGETS), var_name="target", value_name="fixed_exposure_auc")
    matched_long = matched.melt(id_vars=id_columns, value_vars=list(source.TARGETS), var_name="target", value_name="matched40k_auc")
    cls_rows = fixed_long.merge(matched_long, on=["rung", "target"], validate="one_to_one")
    cls_rows["task"] = "classification"
    cls_rows["fixed_total_steps"] = cls_rows.rung * 10_000
    cls_rows["matched40k_total_steps"] = 40_000
    cls_rows = cls_rows[["task", "rung", "target", "fixed_total_steps", "matched40k_total_steps", "fixed_exposure_auc", "matched40k_auc"]]

    cells = pd.concat([nm_rows, cls_rows], ignore_index=True)
    cells["fixed_minus_matched40k"] = cells.fixed_exposure_auc - cells.matched40k_auc
    cells["log_compute_ratio"] = np.log(cells.fixed_total_steps / cells.matched40k_total_steps)
    summaries = []
    for task, part in cells.groupby("task"):
        target_means = part.groupby("target").fixed_minus_matched40k.mean()
        low, high = bootstrap_cluster_ci(target_means)
        equal_compute = part[part.fixed_total_steps.eq(part.matched40k_total_steps)]
        summaries.append(
            {
                "task": task,
                "mean_fixed_minus_matched40k": part.fixed_minus_matched40k.mean(),
                "cluster_bootstrap_low": low,
                "cluster_bootstrap_high": high,
                "mean_at_equal_40k_rung4": equal_compute.fixed_minus_matched40k.mean(),
                "mean_low_compute_rungs1_3": part[part.rung.le(3)].fixed_minus_matched40k.mean(),
                "mean_high_compute_rungs5_8": part[part.rung.ge(5)].fixed_minus_matched40k.mean(),
                "cells": len(part),
            }
        )
    return cells, pd.DataFrame(summaries)


def saturation_summary() -> pd.DataFrame:
    data = pd.read_csv(SATURATION)
    data = data[(data.task.eq("classification")) & data.dataset.isin(source.TARGETS)]
    return data.groupby(["arm", "step"], as_index=False).value_h2.mean().rename(columns={"value_h2": "mean_classification_auc"})


def make_figure(
    frame: pd.DataFrame,
    cv: pd.DataFrame,
    schedule_cells: pd.DataFrame,
    saturation: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.6))

    selected_models = ["max", "second_best", "mean", "softmax", "max_plus_size", "max_plus_rest", "max_plus_size_rest_diversity"]
    pivot = cv[cv.model.isin(selected_models)].pivot(index="model", columns="setting", values="mae").reindex(selected_models)
    x = np.arange(len(selected_models))
    width = .19
    for offset, setting in enumerate(SETTING_LABELS):
        axes[0, 0].bar(
            x + (offset - 1.5) * width, pivot[setting], width,
            label=SETTING_LABELS[setting], color=SETTING_COLORS[setting],
        )
    axes[0, 0].set_xticks(x, [label.replace("_", "\n") for label in selected_models], fontsize=8)
    axes[0, 0].set_ylabel("Leave-one-target-out MAE")
    axes[0, 0].set_title("a  Predictive model tournament", loc="left", fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)

    for setting, part in frame.groupby("setting"):
        residual = part.observed_auc - part.best
        grouped = pd.DataFrame({"n_sources": part.n_sources, "residual": residual}).groupby("n_sources").residual.mean()
        axes[0, 1].plot(
            grouped.index, grouped.values, marker="o", lw=1.8,
            color=SETTING_COLORS[setting], label=SETTING_LABELS[setting],
        )
    axes[0, 1].axhline(0, color="#555555", ls="--", lw=.9)
    axes[0, 1].set_xlabel("Number of source graphs")
    axes[0, 1].set_ylabel("Mixture AUC − best constituent AUC")
    axes[0, 1].set_title("b  Residual mixture-size pattern", loc="left", fontweight="bold")

    schedule_means = schedule_cells.groupby(["task", "rung"]).fixed_minus_matched40k.mean().reset_index()
    for task, part in schedule_means.groupby("task"):
        axes[1, 0].plot(part.rung, part.fixed_minus_matched40k, marker="o", lw=1.9, label=task)
    axes[1, 0].axhline(0, color="#555555", ls="--", lw=.9)
    axes[1, 0].axvline(4, color="#888888", ls=":", lw=1.0)
    axes[1, 0].text(4.08, axes[1, 0].get_ylim()[1] * .88, "equal total compute", fontsize=8, color="#666666")
    axes[1, 0].set_xlabel("Mixture size / rung")
    axes[1, 0].set_ylabel("10k/source − matched-40k AUC")
    axes[1, 0].set_title("c  Same-source-set schedule comparison", loc="left", fontweight="bold")
    axes[1, 0].legend(frameon=False, fontsize=8)

    for arm, part in saturation.groupby("arm"):
        axes[1, 1].plot(part.step + 1, part.mean_classification_auc, marker="o", lw=1.8, label=arm)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("Training updates (+1 for log scale)")
    axes[1, 1].set_ylabel("Mean classification AUC")
    axes[1, 1].set_title("d  Training longer saturates early", loc="left", fontweight="bold")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        ax.grid(axis="y", color="#dddddd", linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("What explains mixture performance?", fontsize=14, fontweight="bold", y=.995)
    fig.tight_layout(rect=(0, 0, 1, .975), w_pad=2.0, h_pad=2.1)
    output = OUT_FIGURES / "mixture_explanation_model_comparison"
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    frame = build_canonical()
    predictions = grouped_predictions(frame)
    cv = summarize_cv(predictions)
    model_tests = paired_model_tests(predictions)
    size_tests, choice_tests = size_and_choice_tests(frame)
    schedule_cells, schedule_summary = schedule_comparison()
    saturation = saturation_summary()

    outputs = {
        "mixture_cells.csv": frame,
        "cross_validated_predictions.csv": predictions,
        "model_comparison.csv": cv,
        "paired_model_tests.csv": model_tests,
        "mixture_size_tests.csv": size_tests,
        "graph_choice_tests.csv": choice_tests,
        "schedule_comparison_cells.csv": schedule_cells,
        "schedule_comparison_summary.csv": schedule_summary,
        "classification_saturation_summary.csv": saturation,
    }
    for filename, table in outputs.items():
        table.to_csv(OUT_DATA / filename, index=False)

    summary = {
        "coverage": frame.groupby("setting").size().to_dict(),
        "best_models_by_mae": {
            setting: part.sort_values("mae").iloc[0][["model", "mae"]].to_dict()
            for setting, part in cv.groupby("setting")
        },
        "analysis_notes": [
            "Prediction comparisons use leave-one-target-out folds over four targets.",
            "Paired uncertainty resamples target-by-order trajectories, not adjacent cells.",
            "Fixed-exposure and classification constituent scores are matrix-reference diagnostics, not same-budget causal estimates.",
            "Total updates and mixture size are collinear within fixed exposure; schedule effects use paired matched-40k controls instead.",
        ],
    }
    (OUT_DATA / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    make_figure(frame, cv, schedule_cells, saturation)
    print(OUT_DATA)


if __name__ == "__main__":
    main()
