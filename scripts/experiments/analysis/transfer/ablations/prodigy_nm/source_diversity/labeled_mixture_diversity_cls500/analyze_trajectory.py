#!/usr/bin/env python3
"""Validate and analyze the 500/750/1000-step mixture-diversity trajectory."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FINAL_CORE_TRANSFER = (
    HERE.parents[7]
    / "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/data"
    / "prodigy_final_core/auc/summary"
    / "single_source_roc_auc_ovr_macro_three_seed_mean.csv"
)
TARGETS = (
    "covid_political", "election2020", "facebook_page_reference",
    "ukr_rus_suspended", "twibot20",
)
STEPS = (500, 750, 1000)
METRICS = ("roc_auc", "accuracy", "f1")


def load_jsonl(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        )
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict], metric: str) -> float:
    return statistics.mean(float(row[metric]) for row in rows)


def endpoint(row: dict) -> str:
    return str(row.get("endpoint", "heldout"))


def cell_key(row: dict, include_step: bool = True) -> tuple:
    key = (str(row["target"]), str(row["model_id"]), endpoint(row))
    return (int(row["training_steps"]), *key) if include_step else key


def validate_step(rows: list[dict], step: int) -> None:
    assert len(rows) == 85, (step, len(rows))
    assert len({cell_key(row) for row in rows}) == 85
    heldout = [row for row in rows if endpoint(row) == "heldout"]
    controls = [row for row in rows if endpoint(row) != "heldout"]
    assert len(heldout) == 75
    assert len(controls) == 10
    assert all(int(row["training_steps"]) == step for row in rows)
    assert all(int(row["eval_episodes"]) == 500 for row in rows)
    for target in TARGETS:
        target_rows = [row for row in heldout if row["target"] == target]
        assert {k: sum(int(row["mixture_size"]) == k for row in target_rows) for k in range(1, 5)} == {
            1: 4, 2: 6, 3: 4, 4: 1,
        }
        assert all(target not in row["donors"] for row in target_rows)
        target_controls = [row for row in controls if row["target"] == target]
        assert {endpoint(row) for row in target_controls} == {"target_only", "all_five"}
        assert all(target in row["donors"] for row in target_controls)
        assert len({row["episode_fingerprint"] for row in target_rows + target_controls}) == 1


def main() -> None:
    heldout_500 = load_jsonl(sorted(DATA.glob("heldout_seed0_shard*.jsonl")))
    controls_500 = load_jsonl(sorted(DATA.glob("controls_seed0_shard*.jsonl")))
    for row in heldout_500:
        row.setdefault("endpoint", "heldout")
        row.setdefault("target_in_training", False)
    trajectory = load_jsonl(sorted((DATA / "trajectory").glob("trajectory_step*_shard*.jsonl")))
    rows = heldout_500 + controls_500 + trajectory
    assert len(rows) == 255, len(rows)
    assert len({cell_key(row) for row in rows}) == 255
    for step in STEPS:
        validate_step([row for row in rows if int(row["training_steps"]) == step], step)

    # Every arm and checkpoint for a target must use the same labeled episodes.
    for target in TARGETS:
        fingerprints = {row["episode_fingerprint"] for row in rows if row["target"] == target}
        assert len(fingerprints) == 1, (target, fingerprints)

    clean_rows = []
    for row in sorted(rows, key=cell_key):
        clean = dict(row)
        clean["endpoint"] = endpoint(row)
        clean["donors"] = ",".join(row["donors"])
        clean_rows.append(clean)
    long_fields = [
        "target", "model_id", "endpoint", "target_in_training", "mixture_size",
        "donors", "training_steps", "training_seed", "eval_episodes",
        "episode_fingerprint", "roc_auc", "accuracy", "f1", "checkpoint",
    ]
    write_csv(DATA / "trajectory_all_results.csv", clean_rows, long_fields)

    summary_rows = []
    target_means: dict[tuple[int, str, int], float] = {}
    for step in STEPS:
        heldout = [
            row for row in rows
            if int(row["training_steps"]) == step and endpoint(row) == "heldout"
        ]
        for k in range(1, 5):
            k_rows = [row for row in heldout if int(row["mixture_size"]) == k]
            summary_rows.append({
                "training_steps": step, "scope": "macro", "target": "all",
                "mixture_size": k, "n": len(k_rows),
                **{metric: mean(k_rows, metric) for metric in METRICS},
            })
        for target in TARGETS:
            for k in range(1, 5):
                k_rows = [
                    row for row in heldout
                    if row["target"] == target and int(row["mixture_size"]) == k
                ]
                target_means[(step, target, k)] = mean(k_rows, "roc_auc")
                summary_rows.append({
                    "training_steps": step, "scope": "target", "target": target,
                    "mixture_size": k, "n": len(k_rows),
                    **{metric: mean(k_rows, metric) for metric in METRICS},
                })
    write_csv(DATA / "trajectory_summary.csv", summary_rows, list(summary_rows[0]))

    endpoint_rows = []
    for step in STEPS:
        step_rows = [row for row in rows if int(row["training_steps"]) == step]
        for target in TARGETS:
            k4 = next(
                row for row in step_rows
                if row["target"] == target and endpoint(row) == "heldout"
                and int(row["mixture_size"]) == 4
            )
            target_only = next(
                row for row in step_rows
                if row["target"] == target and endpoint(row) == "target_only"
            )
            all_five = next(
                row for row in step_rows
                if row["target"] == target and endpoint(row) == "all_five"
            )
            endpoint_rows.append({
                "training_steps": step, "target": target,
                "heldout_k4_auc": float(k4["roc_auc"]),
                "target_only_auc": float(target_only["roc_auc"]),
                "all_five_auc": float(all_five["roc_auc"]),
                "all_five_minus_k4": float(all_five["roc_auc"]) - float(k4["roc_auc"]),
                "target_only_minus_k4": float(target_only["roc_auc"]) - float(k4["roc_auc"]),
            })
    write_csv(DATA / "trajectory_endpoint_controls.csv", endpoint_rows, list(endpoint_rows[0]))

    by_step_cell = {(int(row["training_steps"]), cell_key(row, False)): row for row in rows}
    delta_rows = []
    for before, after in ((500, 750), (750, 1000)):
        for key in sorted({cell_key(row, False) for row in rows}):
            left, right = by_step_cell[(before, key)], by_step_cell[(after, key)]
            delta_rows.append({
                "from_step": before, "to_step": after,
                "target": key[0], "model_id": key[1], "endpoint": key[2],
                "mixture_size": int(right["mixture_size"]),
                "donors": ",".join(right["donors"]),
                **{
                    f"{metric}_delta": float(right[metric]) - float(left[metric])
                    for metric in METRICS
                },
            })
    write_csv(DATA / "trajectory_cell_deltas.csv", delta_rows, list(delta_rows[0]))

    convergence_rows = []
    for before, after in ((500, 750), (750, 1000)):
        period = [row for row in delta_rows if row["from_step"] == before]
        scopes = [("all", "all", period), ("heldout", "all", [r for r in period if r["endpoint"] == "heldout"])]
        scopes.extend(
            ("heldout_target", target, [
                row for row in period
                if row["endpoint"] == "heldout" and row["target"] == target
            ])
            for target in TARGETS
        )
        for scope, target, scoped in scopes:
            auc_deltas = [float(row["roc_auc_delta"]) for row in scoped]
            abs_deltas = [abs(value) for value in auc_deltas]
            convergence_rows.append({
                "from_step": before, "to_step": after, "scope": scope,
                "target": target, "n": len(scoped),
                "mean_auc_delta": statistics.mean(auc_deltas),
                "median_auc_delta": statistics.median(auc_deltas),
                "mean_abs_auc_delta": statistics.mean(abs_deltas),
                "median_abs_auc_delta": statistics.median(abs_deltas),
                "max_abs_auc_delta": max(abs_deltas),
                "fraction_abs_le_0_005": sum(value <= 0.005 for value in abs_deltas) / len(abs_deltas),
                "fraction_abs_le_0_01": sum(value <= 0.01 for value in abs_deltas) / len(abs_deltas),
            })
    write_csv(DATA / "trajectory_convergence.csv", convergence_rows, list(convergence_rows[0]))

    final_period = [row for row in delta_rows if row["from_step"] == 750]
    model_convergence_rows = []
    for model_id in sorted({row["model_id"] for row in final_period}):
        model_rows = [row for row in final_period if row["model_id"] == model_id]
        abs_deltas = [abs(float(row["roc_auc_delta"])) for row in model_rows]
        model_convergence_rows.append({
            "model_id": model_id, "n_eval_cells": len(model_rows),
            "mean_abs_auc_delta": statistics.mean(abs_deltas),
            "max_abs_auc_delta": max(abs_deltas),
            "n_cells_abs_gt_0_01": sum(value > 0.01 for value in abs_deltas),
            "continue_recommended": max(abs_deltas) > 0.01,
        })
    model_convergence_rows.sort(key=lambda row: (-row["max_abs_auc_delta"], row["model_id"]))
    write_csv(
        DATA / "trajectory_model_convergence.csv",
        model_convergence_rows,
        list(model_convergence_rows[0]),
    )

    ks = np.array([1, 2, 3, 4], dtype=float)
    slope_rows = []
    step_rows = []
    macro_curves = {}
    for step in STEPS:
        macro = [
            next(
                float(row["roc_auc"]) for row in summary_rows
                if row["training_steps"] == step and row["scope"] == "macro"
                and row["mixture_size"] == k
            )
            for k in range(1, 5)
        ]
        macro_curves[step] = macro
        macro_slope = float(np.polyfit(ks, macro, 1)[0])
        slope_rows.append({
            "training_steps": step, "target": "all", "slope_per_source": macro_slope,
            "k1_auc": macro[0], "k4_auc": macro[-1], "k4_minus_k1": macro[-1] - macro[0],
        })
        for target in TARGETS:
            curve = [target_means[(step, target, k)] for k in range(1, 5)]
            slope_rows.append({
                "training_steps": step, "target": target,
                "slope_per_source": float(np.polyfit(ks, curve, 1)[0]),
                "k1_auc": curve[0], "k4_auc": curve[-1], "k4_minus_k1": curve[-1] - curve[0],
            })
        endpoint_step = [row for row in endpoint_rows if row["training_steps"] == step]
        step_rows.append({
            "training_steps": step, "macro_k1_auc": macro[0], "macro_k4_auc": macro[-1],
            "macro_k4_minus_k1": macro[-1] - macro[0],
            "macro_slope_per_source": macro_slope,
            **{
                f"mean_{field}": statistics.mean(float(row[field]) for row in endpoint_step)
                for field in (
                    "heldout_k4_auc", "target_only_auc", "all_five_auc",
                    "all_five_minus_k4", "target_only_minus_k4",
                )
            },
        })
    write_csv(DATA / "trajectory_slopes.csv", slope_rows, list(slope_rows[0]))
    write_csv(DATA / "trajectory_step_summary.csv", step_rows, list(step_rows[0]))

    marginal_rows = []
    marginal_edge_rows = []
    for step in STEPS:
        heldout = [
            row for row in rows
            if int(row["training_steps"]) == step and endpoint(row) == "heldout"
        ]
        by_subset = {
            (row["target"], frozenset(row["donors"])): float(row["roc_auc"])
            for row in heldout
        }
        for target in TARGETS:
            donors = sorted(set().union(*(
                set(row["donors"]) for row in heldout if row["target"] == target
            )))
            for donor in donors:
                deltas = []
                for (row_target, subset), value in by_subset.items():
                    expanded = frozenset(set(subset) | {donor})
                    if row_target == target and donor not in subset and (target, expanded) in by_subset:
                        delta = by_subset[(target, expanded)] - value
                        deltas.append(delta)
                        marginal_edge_rows.append({
                            "training_steps": step,
                            "target": target,
                            "added_donor": donor,
                            "base_donors": ",".join(sorted(subset)),
                            "expanded_donors": ",".join(sorted(expanded)),
                            "roc_auc_delta": delta,
                        })
                assert len(deltas) == 7
                marginal_rows.append({
                    "training_steps": step, "target": target, "added_donor": donor,
                    "n_subset_edges": len(deltas), "mean_auc_delta": statistics.mean(deltas),
                    "min_auc_delta": min(deltas), "max_auc_delta": max(deltas),
                })
    write_csv(DATA / "trajectory_marginal_donor_effects.csv", marginal_rows, list(marginal_rows[0]))
    write_csv(
        DATA / "trajectory_marginal_donor_edge_effects.csv",
        marginal_edge_rows,
        list(marginal_edge_rows[0]),
    )

    figures = HERE / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    # The k=1 held-out arms plus the target-only controls form an exact 5x5
    # single-source transfer matrix under the 1k mixture-study protocol.
    single_source_rows = [
        row for row in rows
        if int(row["training_steps"]) == 1000
        and int(row["mixture_size"]) == 1
        and endpoint(row) in {"heldout", "target_only"}
    ]
    assert len(single_source_rows) == len(TARGETS) ** 2
    single_source_by_cell = {
        (row["donors"][0], row["target"]): row
        for row in single_source_rows
    }
    assert len(single_source_by_cell) == len(TARGETS) ** 2
    single_source_long_rows = []
    single_source_wide_rows = []
    single_source_matrix = np.empty((len(TARGETS), len(TARGETS)))
    for source_index, source in enumerate(TARGETS):
        wide_row = {"training_graph": source}
        for target_index_value, target in enumerate(TARGETS):
            row = single_source_by_cell[(source, target)]
            value = float(row["roc_auc"])
            single_source_matrix[source_index, target_index_value] = value
            wide_row[target] = value
            single_source_long_rows.append({
                "training_graph": source,
                "evaluation_graph": target,
                "endpoint": endpoint(row),
                "training_steps": 1000,
                "training_seed": int(row["training_seed"]),
                "eval_episodes": int(row["eval_episodes"]),
                "roc_auc": value,
                "accuracy": float(row["accuracy"]),
                "f1": float(row["f1"]),
            })
        single_source_wide_rows.append(wide_row)
    write_csv(
        DATA / "single_source_transfer_matrix_1000_long.csv",
        single_source_long_rows,
        list(single_source_long_rows[0]),
    )
    write_csv(
        DATA / "single_source_transfer_matrix_1000.csv",
        single_source_wide_rows,
        ["training_graph", *TARGETS],
    )

    labels = [target.replace("_", "\n") for target in TARGETS]
    fig, axis = plt.subplots(figsize=(9, 7))
    single_source_image = axis.imshow(
        single_source_matrix, cmap="viridis", vmin=0.45, vmax=1.0
    )
    axis.set_xticks(range(len(TARGETS)), labels=labels)
    axis.set_yticks(range(len(TARGETS)), labels=labels)
    axis.set_xlabel("Evaluation graph")
    axis.set_ylabel("Single training graph")
    axis.set_title("Single-source transfer at 1,000 steps")
    for source_index in range(len(TARGETS)):
        for target_index_value in range(len(TARGETS)):
            value = single_source_matrix[source_index, target_index_value]
            axis.text(
                target_index_value, source_index, f"{value:.3f}",
                ha="center", va="center",
                color="white" if value < 0.72 else "black",
                fontsize=9,
                fontweight="bold" if source_index == target_index_value else "normal",
            )
    colorbar = fig.colorbar(single_source_image, ax=axis, shrink=0.82)
    colorbar.set_label("ROC-AUC")
    fig.text(
        0.5, 0.01,
        "Diagonal: target-only training. Off-diagonal: held-out transfer. "
        "Seed 0; 500 paired 10-shot CLS episodes.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figures / "single_source_transfer_matrix_1000.png", dpi=180)
    plt.close(fig)

    final_marginal_edges = [
        row for row in marginal_edge_rows if row["training_steps"] == 1000
    ]
    donor_effects = [
        [
            row["roc_auc_delta"] for row in final_marginal_edges
            if row["added_donor"] == donor
        ]
        for donor in TARGETS
    ]
    assert all(len(values) == 28 for values in donor_effects)
    fig, axis = plt.subplots(figsize=(11, 5.5))
    boxes = axis.boxplot(
        donor_effects,
        tick_labels=[target.replace("_", "\n") for target in TARGETS],
        showmeans=True,
        patch_artist=True,
        meanprops={"marker": "D", "markerfacecolor": "C3", "markeredgecolor": "C3"},
    )
    for box in boxes["boxes"]:
        box.set(facecolor="C2", alpha=0.25)
    rng = np.random.default_rng(0)
    for position, values in enumerate(donor_effects, start=1):
        axis.scatter(
            position + rng.uniform(-0.09, 0.09, len(values)),
            values,
            s=16,
            color="C2",
            alpha=0.55,
            zorder=2,
        )
    axis.axhline(0, color="black", linewidth=1)
    axis.set_xlabel("Graph added to the training mixture")
    axis.set_ylabel("Change in held-out ROC-AUC")
    axis.set_title("Marginal effect of adding each graph at 1,000 steps (n=28 per graph)")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "marginal_donor_effect_boxplot_1000.png", dpi=180)
    plt.close(fig)

    final_marginal_means = [
        row for row in marginal_rows if row["training_steps"] == 1000
    ]
    effect_matrix = np.full((len(TARGETS), len(TARGETS)), np.nan)
    target_index = {target: index for index, target in enumerate(TARGETS)}
    for row in final_marginal_means:
        effect_matrix[
            target_index[row["added_donor"]], target_index[row["target"]]
        ] = row["mean_auc_delta"]
    max_abs = float(np.nanmax(np.abs(effect_matrix)))
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("0.9")
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    fig, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(effect_matrix, cmap=cmap, norm=norm)
    axis.set_xticks(range(len(TARGETS)), labels=labels)
    axis.set_yticks(range(len(TARGETS)), labels=labels)
    axis.set_xlabel("Held-out evaluation graph")
    axis.set_ylabel("Graph added to the training mixture")
    axis.set_title("Mean marginal effect of adding each graph at 1,000 steps")
    for row_index in range(len(TARGETS)):
        for column_index in range(len(TARGETS)):
            value = effect_matrix[row_index, column_index]
            if np.isnan(value):
                annotation, color = "—", "0.35"
            else:
                annotation = f"{value:+.4f}"
                color = "white" if abs(value) > 0.55 * max_abs else "black"
            axis.text(
                column_index, row_index, annotation,
                ha="center", va="center", color=color, fontsize=9,
            )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label("Mean change in held-out ROC-AUC")
    fig.tight_layout()
    fig.savefig(figures / "marginal_donor_effect_heatmap_1000.png", dpi=180)
    plt.close(fig)

    # Same-protocol comparison: both standalone transfer and marginal mixture
    # value come from the 1k, seed-0, paired 10-shot experiment above.
    same_protocol_rows = []
    for added_donor in TARGETS:
        for target in TARGETS:
            if added_donor == target:
                continue
            same_protocol_rows.append({
                "added_donor": added_donor,
                "target": target,
                "single_source_auc_1000": single_source_matrix[
                    target_index[added_donor], target_index[target]
                ],
                "mean_marginal_auc_delta_1000": effect_matrix[
                    target_index[added_donor], target_index[target]
                ],
            })
    assert len(same_protocol_rows) == 20
    write_csv(
        DATA / "marginal_vs_single_source_1000_same_protocol.csv",
        same_protocol_rows,
        list(same_protocol_rows[0]),
    )

    same_protocol_transfer_values = np.array([
        row["single_source_auc_1000"] for row in same_protocol_rows
    ])
    same_protocol_marginal_values = np.array([
        row["mean_marginal_auc_delta_1000"] for row in same_protocol_rows
    ])
    same_protocol_pearson = pearsonr(
        same_protocol_transfer_values, same_protocol_marginal_values
    )
    same_protocol_spearman = spearmanr(
        same_protocol_transfer_values, same_protocol_marginal_values
    )
    same_protocol_within_target_rhos = []
    same_protocol_best_donor_matches = 0
    for target in TARGETS:
        target_rows = [row for row in same_protocol_rows if row["target"] == target]
        same_protocol_within_target_rhos.append(spearmanr(
            [row["single_source_auc_1000"] for row in target_rows],
            [row["mean_marginal_auc_delta_1000"] for row in target_rows],
        ).statistic)
        best_single_source = max(
            target_rows, key=lambda row: row["single_source_auc_1000"]
        )["added_donor"]
        best_marginal = max(
            target_rows, key=lambda row: row["mean_marginal_auc_delta_1000"]
        )["added_donor"]
        same_protocol_best_donor_matches += best_single_source == best_marginal
    same_protocol_mean_within_target_rho = statistics.mean(
        same_protocol_within_target_rhos
    )

    single_source_off_diagonal = single_source_matrix.copy()
    np.fill_diagonal(single_source_off_diagonal, np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
    transfer_cmap = plt.get_cmap("viridis").copy()
    transfer_cmap.set_bad("0.9")
    transfer_image = axes[0].imshow(
        single_source_off_diagonal, cmap=transfer_cmap, vmin=0.45, vmax=1.0
    )
    marginal_image = axes[1].imshow(effect_matrix, cmap=cmap, norm=norm)
    for axis in axes[:2]:
        axis.set_xticks(range(len(TARGETS)), labels=labels)
        axis.set_yticks(range(len(TARGETS)), labels=labels)
        axis.set_xlabel("Held-out evaluation graph")
    axes[0].set_ylabel("Single training graph")
    axes[1].set_ylabel("Graph added to the mixture")
    axes[0].set_title("Single-source transfer AUC at 1k")
    axes[1].set_title("Mean marginal addition effect at 1k")
    for row_index in range(len(TARGETS)):
        for column_index in range(len(TARGETS)):
            transfer_value = single_source_off_diagonal[row_index, column_index]
            marginal_value = effect_matrix[row_index, column_index]
            if np.isnan(transfer_value):
                axes[0].text(
                    column_index, row_index, "—",
                    ha="center", va="center", color="0.35",
                )
                axes[1].text(
                    column_index, row_index, "—",
                    ha="center", va="center", color="0.35",
                )
            else:
                axes[0].text(
                    column_index, row_index, f"{transfer_value:.3f}",
                    ha="center", va="center",
                    color="white" if transfer_value < 0.72 else "black",
                    fontsize=8,
                )
                axes[1].text(
                    column_index, row_index, f"{marginal_value:+.3f}",
                    ha="center", va="center",
                    color="white" if abs(marginal_value) > 0.55 * max_abs else "black",
                    fontsize=8,
                )
    fig.colorbar(transfer_image, ax=axes[0], shrink=0.75, label="ROC-AUC")
    fig.colorbar(
        marginal_image, ax=axes[1], shrink=0.75,
        label="Mean change in ROC-AUC",
    )

    same_protocol_donor_colors = dict(zip(TARGETS, plt.get_cmap("tab10").colors))
    for donor in TARGETS:
        donor_rows = [
            row for row in same_protocol_rows if row["added_donor"] == donor
        ]
        axes[2].scatter(
            [row["single_source_auc_1000"] for row in donor_rows],
            [row["mean_marginal_auc_delta_1000"] for row in donor_rows],
            label=donor.replace("_", " "),
            color=same_protocol_donor_colors[donor],
            s=42,
            alpha=0.85,
        )
    fit = np.polyfit(
        same_protocol_transfer_values, same_protocol_marginal_values, 1
    )
    fit_x = np.linspace(
        same_protocol_transfer_values.min(), same_protocol_transfer_values.max(), 100
    )
    axes[2].plot(fit_x, np.polyval(fit, fit_x), color="0.25", linewidth=1)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_xlabel("1k single-source transfer ROC-AUC")
    axes[2].set_ylabel("Mean marginal addition effect")
    axes[2].set_title("Off-diagonal cells")
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=7, frameon=False)
    axes[2].text(
        0.03, 0.97,
        f"Pearson r={same_protocol_pearson.statistic:.2f}\n"
        f"Spearman ρ={same_protocol_spearman.statistic:.2f}\n"
        f"Mean within-target ρ={same_protocol_mean_within_target_rho:.2f}\n"
        f"Best-donor match={same_protocol_best_donor_matches}/5",
        transform=axes[2].transAxes,
        ha="left", va="top", fontsize=9,
    )
    fig.suptitle("Standalone transfer versus marginal value in a diverse mixture")
    fig.text(
        0.5, 0.01,
        "Both matrices: 1k training steps, seed 0, 500 paired 10-shot CLS episodes.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(
        figures / "marginal_vs_single_source_1000_same_protocol.png", dpi=180
    )
    plt.close(fig)

    with FINAL_CORE_TRANSFER.open(newline="", encoding="utf-8") as handle:
        final_core_auc = {
            (row["model_source"], target): float(row[target])
            for row in csv.DictReader(handle)
            for target in TARGETS
        }
    comparison_rows = []
    for added_donor in TARGETS:
        for target in TARGETS:
            if added_donor == target:
                continue
            comparison_rows.append({
                "added_donor": added_donor,
                "target": target,
                "final_core_2500_single_source_auc": final_core_auc[(added_donor, target)],
                "mean_marginal_auc_delta_1000": effect_matrix[
                    target_index[added_donor], target_index[target]
                ],
            })
    assert len(comparison_rows) == 20
    write_csv(
        DATA / "marginal_vs_final_core_2500_transfer_1000.csv",
        comparison_rows,
        list(comparison_rows[0]),
    )

    final_core_matrix = np.full((len(TARGETS), len(TARGETS)), np.nan)
    for row in comparison_rows:
        final_core_matrix[
            target_index[row["added_donor"]], target_index[row["target"]]
        ] = row["final_core_2500_single_source_auc"]
    final_core_values = np.array([
        row["final_core_2500_single_source_auc"] for row in comparison_rows
    ])
    marginal_values = np.array([
        row["mean_marginal_auc_delta_1000"] for row in comparison_rows
    ])
    pearson = pearsonr(final_core_values, marginal_values)
    spearman = spearmanr(final_core_values, marginal_values)
    within_target_rhos = []
    best_donor_matches = 0
    for target in TARGETS:
        target_rows = [row for row in comparison_rows if row["target"] == target]
        within_target_rhos.append(spearmanr(
            [row["final_core_2500_single_source_auc"] for row in target_rows],
            [row["mean_marginal_auc_delta_1000"] for row in target_rows],
        ).statistic)
        best_single_source = max(
            target_rows, key=lambda row: row["final_core_2500_single_source_auc"]
        )["added_donor"]
        best_marginal = max(
            target_rows, key=lambda row: row["mean_marginal_auc_delta_1000"]
        )["added_donor"]
        best_donor_matches += best_single_source == best_marginal
    mean_within_target_rho = statistics.mean(within_target_rhos)
    if abs(mean_within_target_rho) < 0.005:
        mean_within_target_rho = 0.0

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
    final_core_cmap = plt.get_cmap("viridis").copy()
    final_core_cmap.set_bad("0.9")
    final_core_image = axes[0].imshow(
        final_core_matrix, cmap=final_core_cmap, vmin=0.5, vmax=1.0
    )
    marginal_image = axes[1].imshow(effect_matrix, cmap=cmap, norm=norm)
    for axis in axes[:2]:
        axis.set_xticks(range(len(TARGETS)), labels=labels)
        axis.set_yticks(range(len(TARGETS)), labels=labels)
        axis.set_xlabel("Held-out evaluation graph")
    axes[0].set_ylabel("Single training graph")
    axes[1].set_ylabel("Graph added to the mixture")
    axes[0].set_title("Final-core single-source transfer AUC (2.5k, 3-seed mean)")
    axes[1].set_title("Mean marginal addition effect at 1k")
    for row_index in range(len(TARGETS)):
        for column_index in range(len(TARGETS)):
            final_core_value = final_core_matrix[row_index, column_index]
            marginal_value = effect_matrix[row_index, column_index]
            if np.isnan(final_core_value):
                axes[0].text(
                    column_index, row_index, "—",
                    ha="center", va="center", color="0.35",
                )
                axes[1].text(
                    column_index, row_index, "—",
                    ha="center", va="center", color="0.35",
                )
            else:
                axes[0].text(
                    column_index, row_index, f"{final_core_value:.3f}",
                    ha="center", va="center",
                    color="white" if final_core_value < 0.72 else "black",
                    fontsize=8,
                )
                axes[1].text(
                    column_index, row_index, f"{marginal_value:+.3f}",
                    ha="center", va="center",
                    color="white" if abs(marginal_value) > 0.55 * max_abs else "black",
                    fontsize=8,
                )
    fig.colorbar(final_core_image, ax=axes[0], shrink=0.75, label="ROC-AUC")
    fig.colorbar(
        marginal_image, ax=axes[1], shrink=0.75,
        label="Mean change in ROC-AUC",
    )

    donor_colors = dict(zip(TARGETS, plt.get_cmap("tab10").colors))
    for donor in TARGETS:
        donor_rows = [row for row in comparison_rows if row["added_donor"] == donor]
        axes[2].scatter(
            [row["final_core_2500_single_source_auc"] for row in donor_rows],
            [row["mean_marginal_auc_delta_1000"] for row in donor_rows],
            label=donor.replace("_", " "),
            color=donor_colors[donor],
            s=42,
            alpha=0.85,
        )
    fit = np.polyfit(final_core_values, marginal_values, 1)
    fit_x = np.linspace(final_core_values.min(), final_core_values.max(), 100)
    axes[2].plot(fit_x, np.polyval(fit, fit_x), color="0.25", linewidth=1)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_xlabel("Final-core 2.5k single-source transfer ROC-AUC")
    axes[2].set_ylabel("Mean marginal addition effect")
    axes[2].set_title("Off-diagonal cells")
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=7, frameon=False)
    axes[2].text(
        0.03, 0.97,
        f"Pearson r={pearson.statistic:.2f}\n"
        f"Spearman ρ={spearman.statistic:.2f}\n"
        f"Mean within-target ρ={mean_within_target_rho:.2f}\n"
        f"Best-donor match={best_donor_matches}/5",
        transform=axes[2].transAxes,
        ha="left", va="top", fontsize=9,
    )
    fig.suptitle("Standalone transfer versus marginal value in a diverse mixture")
    fig.text(
        0.5, 0.01,
        "Final-core matrix: 2.5k steps, three-seed mean, 30-way/3-shot. "
        "Marginal matrix: 1k-step labeled CLS, seed 0, 10-shot. "
        "Compare patterns, not raw scales.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(figures / "marginal_vs_final_core_2500_transfer_1000.png", dpi=180)
    plt.close(fig)

    panels = [("Macro mean", None), *[(target.replace("_", " "), target) for target in TARGETS]]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for axis, (title, target) in zip(axes.flat, panels):
        for step in STEPS:
            curve = macro_curves[step] if target is None else [
                target_means[(step, target, k)] for k in range(1, 5)
            ]
            axis.plot(ks, curve, marker="o", label=f"{step} steps")
        axis.set_title(title)
        axis.set_xticks(ks)
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Number of held-in pretraining graphs")
    for axis in axes[:, 0]:
        axis.set_ylabel("Held-out ROC-AUC")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "mixture_diversity_trajectory.png", dpi=180)
    plt.close(fig)

    # Explicit 1k-only averaged view, retained alongside the all-checkpoint trajectory.
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for axis, (title, target) in zip(axes.flat, panels):
        curve = macro_curves[1000] if target is None else [
            target_means[(1000, target, k)] for k in range(1, 5)
        ]
        axis.plot(ks, curve, marker="o", color="C2")
        axis.set_title(title)
        axis.set_xticks(ks)
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Number of held-in pretraining graphs")
    for axis in axes[:, 0]:
        axis.set_ylabel("Held-out ROC-AUC")
    fig.suptitle("1,000-step mixture-diversity averages")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figures / "mixture_diversity_trajectory_averaged_1000.png", dpi=180)
    plt.close(fig)

    # Explicit 1k-only raw view containing all 75 held-out mixture cells.
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for axis, target in zip(axes.flat[:5], TARGETS):
        target_rows = [
            row for row in rows
            if row["target"] == target
            and endpoint(row) == "heldout"
            and int(row["training_steps"]) == 1000
        ]
        model_jitter = {}
        for k in range(1, 5):
            model_ids = sorted({
                row["model_id"] for row in target_rows
                if int(row["mixture_size"]) == k
            })
            jitters = np.linspace(-0.12, 0.12, len(model_ids)) if len(model_ids) > 1 else [0.0]
            model_jitter.update(dict(zip(model_ids, jitters)))
        xs = [
            int(row["mixture_size"]) + model_jitter[row["model_id"]]
            for row in target_rows
        ]
        axis.scatter(
            xs, [float(row["roc_auc"]) for row in target_rows],
            s=28, alpha=0.85, color="C2", zorder=2,
        )
        axis.set_title(target.replace("_", " "))
        axis.set_xticks(ks)
        axis.grid(alpha=0.2)
    axes.flat[5].axis("off")
    axes.flat[5].text(
        0.5, 0.5, "Each point is one exact donor subset.",
        ha="center", va="center", transform=axes.flat[5].transAxes,
    )
    for axis in axes[-1, :2]:
        axis.set_xlabel("Number of held-in pretraining graphs")
    for axis in axes[:, 0]:
        axis.set_ylabel("Held-out ROC-AUC")
    fig.suptitle("All 1,000-step held-out mixtures")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figures / "mixture_diversity_trajectory_all_points.png", dpi=180)
    plt.close(fig)

    # Five raw 1k-cell views, each filtered to mixtures containing one graph. The
    # filter graph cannot also be a target because held-out targets are absent from
    # their training mixtures.
    for included_donor in TARGETS:
        donor_targets = [target for target in TARGETS if target != included_donor]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
        for axis, target in zip(axes.flat, donor_targets):
            target_rows = [
                row for row in rows
                if row["target"] == target
                and endpoint(row) == "heldout"
                and int(row["training_steps"]) == 1000
                and included_donor in row["donors"]
            ]
            model_jitter = {}
            for k in range(1, 5):
                model_ids = sorted({
                    row["model_id"] for row in target_rows
                    if int(row["mixture_size"]) == k
                })
                jitters = np.linspace(-0.12, 0.12, len(model_ids)) if len(model_ids) > 1 else [0.0]
                model_jitter.update(dict(zip(model_ids, jitters)))
            xs = [
                int(row["mixture_size"]) + model_jitter[row["model_id"]]
                for row in target_rows
            ]
            axis.scatter(
                xs, [float(row["roc_auc"]) for row in target_rows],
                s=28, alpha=0.85, color="C2", label="1,000 steps", zorder=2,
            )
            axis.set_title(target.replace("_", " "))
            axis.set_xticks(ks)
            axis.grid(alpha=0.2)
        for axis in axes[-1]:
            axis.set_xlabel("Number of held-in pretraining graphs")
        for axis in axes[:, 0]:
            axis.set_ylabel("Held-out ROC-AUC")
        fig.suptitle(
            f"1,000-step mixtures containing {included_donor.replace('_', ' ')}"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(
            figures / f"mixture_diversity_trajectory_contains_{included_donor}.png",
            dpi=180,
        )
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for field, label in (
        ("mean_heldout_k4_auc", "4-source held out"),
        ("mean_target_only_auc", "target only"),
        ("mean_all_five_auc", "all five"),
    ):
        axes[0].plot(STEPS, [row[field] for row in step_rows], marker="o", label=label)
    axes[0].set(xlabel="Training steps", ylabel="Macro ROC-AUC", xticks=STEPS)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)
    final_deltas = [row for row in delta_rows if row["from_step"] == 750 and row["endpoint"] == "heldout"]
    grouped = [[row["roc_auc_delta"] for row in final_deltas if row["target"] == target] for target in TARGETS]
    axes[1].boxplot(grouped, tick_labels=[target.replace("_", "\n") for target in TARGETS], showfliers=True)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("Held-out ROC-AUC change, 750→1000")
    axes[1].tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(figures / "endpoint_and_convergence_trajectory.png", dpi=180)
    plt.close(fig)

    s = {row["training_steps"]: row for row in step_rows}
    final_conv = next(
        row for row in convergence_rows
        if row["from_step"] == 750 and row["scope"] == "heldout"
    )
    final_slopes = {
        row["target"]: row["slope_per_source"]
        for row in slope_rows if row["training_steps"] == 1000 and row["target"] != "all"
    }
    material = sum(
        abs(float(row["roc_auc_delta"])) > 0.01
        for row in delta_rows
        if row["from_step"] == 750 and row["endpoint"] == "heldout"
    )
    continuation_models = sum(row["continue_recommended"] for row in model_convergence_rows)
    findings = [
        "# Results", "", "## Answer", "",
        "**Mixture diversity improves held-out CLS performance under fixed compute, and the "
        "positive macro relationship remains after continuing every model to 1,000 steps. "
        "The size of the effect is target- and donor-dependent.**", "",
        *(f"- At {step} steps, macro held-out ROC-AUC moves from "
          f"{s[step]['macro_k1_auc']:.4f} for one source to {s[step]['macro_k4_auc']:.4f} "
          f"for four sources ({s[step]['macro_k4_minus_k1']:+.4f}; slope "
          f"{s[step]['macro_slope_per_source']:+.4f} per added source)." for step in STEPS),
        "", "At 1,000 steps, target-specific diversity slopes are: "
        + ", ".join(f"`{target}` {value:+.4f}" for target, value in final_slopes.items()) + ".",
        "", "## Convergence check", "",
        f"From 750 to 1,000 steps, the 75 held-out cells change by "
        f"{final_conv['mean_auc_delta']:+.4f} ROC-AUC on average; median absolute change is "
        f"{final_conv['median_abs_auc_delta']:.4f}, mean absolute change is "
        f"{final_conv['mean_abs_auc_delta']:.4f}, and {material}/75 cells move by more than 0.01.",
        f"A strict model-level rule that continues any model with at least one evaluation cell "
        f"moving by more than 0.01 selects {continuation_models}/31 models; they are listed in "
        "`data/trajectory_model_convergence.csv`.",
        "", "This is a checkpoint-stability diagnostic, not proof of asymptotic convergence. "
        "The fixed-compute diversity result is stable across checkpoints, but a fully "
        "convergence-controlled comparison requires continuing the selected models.",
        "", "## Endpoint controls", "",
        f"At 1,000 steps, macro ROC-AUC is {s[1000]['mean_heldout_k4_auc']:.4f} for the "
        f"four-source held-out model, {s[1000]['mean_target_only_auc']:.4f} for target-only "
        f"training, and {s[1000]['mean_all_five_auc']:.4f} for all-five training. Adding the "
        f"target to the four-source mixture changes the macro mean by "
        f"{s[1000]['mean_all_five_minus_k4']:+.4f}.",
        "", "## Single-source transfer matrix at 1k", "",
        "The five k=1 models, evaluated on all five targets, form a complete 5x5 "
        "single-source transfer matrix at 1,000 steps. Its 20 off-diagonal cells are "
        "held-out transfer evaluations and its five diagonal cells are the target-only "
        "controls. The wide and long tables are saved as "
        "`data/single_source_transfer_matrix_1000.csv` and "
        "`data/single_source_transfer_matrix_1000_long.csv`; the annotated heatmap is "
        "`figures/single_source_transfer_matrix_1000.png`.",
        "", "## Same-protocol standalone transfer versus marginal value", "",
        "With both quantities measured at 1k under the same seed and evaluation protocol, "
        f"the 20 off-diagonal cells have Pearson `r={same_protocol_pearson.statistic:.2f}` "
        f"and overall Spearman `rho={same_protocol_spearman.statistic:.2f}`. The mean "
        f"within-target donor-rank correlation is `{same_protocol_mean_within_target_rho:.2f}`, "
        "showing that standalone strength is substantially more informative for choosing "
        "among donors once target difficulty is controlled. However, the top standalone "
        f"donor is also the top marginal donor for only {same_protocol_best_donor_matches}/5 "
        "targets, so marginal mixture value is still not a max rule.",
        "", "## Comparison with final-core 2.5k single-source transfer", "",
        "After restricting the final-core 2.5k three-seed single-source transfer matrix to "
        "these five graphs, standalone transfer strength is not a reliable predictor of marginal "
        "value in a mixture. Across the 20 directed off-diagonal cells, Pearson "
        f"`r={pearson.statistic:.2f}` and Spearman `rho={spearman.statistic:.2f}`. These are "
        "descriptive coefficients because cells share donors and targets. The mean within-target "
        f"donor-rank correlation is `{mean_within_target_rho:.2f}`, and the best "
        f"standalone donor is also the best marginal donor for {best_donor_matches}/5 targets. "
        "TwiBot20 is strong under both views; Facebook transfers well as a singleton here but "
        "is slightly harmful on average when added to an existing mixture.",
        "", "This is still a pattern comparison rather than a controlled numerical contrast: "
        "the final-core matrix is a three-seed mean at 2.5k steps with 30-way/3-shot evaluation, "
        "whereas the marginal matrix uses seed 0 at 1k steps with 10-shot evaluation.",
        "", "## Scope", "",
        "All arms use training seed 0 and 500 paired 10-shot CLS evaluation episodes. "
        "Fingerprints are identical within each target across all arms and checkpoints. "
        "The experiment holds total optimizer steps fixed within each checkpoint; it does not "
        "hold per-source exposure fixed and does not estimate training-seed uncertainty.",
    ]
    (HERE / "RESULTS.md").write_text("\n".join(findings) + "\n", encoding="utf-8")
    print("validated 255 cells: 85 at each of 500, 750, and 1000 steps")


if __name__ == "__main__":
    main()
