#!/usr/bin/env python3
"""Analyze the fixed-exposure, fair-two-hop NM ladder.

The raw evidence is a 120-row extraction of Tucker ``metrics_test_step0.json``
files: eight Order-A models and seven Order-C models, each evaluated on all eight
source graphs.  Order C rung 8 deliberately reuses the Order-A all-eight model.
This script expands that shared artifact into two *logical* rung-8 rows while
retaining its model/evidence provenance.

Outputs:

* ``data/logical_results.csv``: 128 logical order/rung/test-graph cells;
* ``data/rung_summary.csv``: all-graph, in-mixture, and held-out means;
* ``data/adjacent_deltas.csv``: every cell's change after each source addition;
* ``data/entry_jumps.csv``: the 14 measurable OOD-to-ID entry events;
* ``data/comparison_to_matched40k_h1_orderA.csv``: cross-protocol fixed-exposure
  fair-two-hop vs matched-40k one-hop Order-A cells;
* ``data/comparison_to_matched40k_h2_orderA.csv``: controlled fixed-exposure vs
  matched-40k fair-two-hop Order-A cells;
* ``data/summary.json``: headline statistics used by ``FINDINGS.md``; and
* ``figures/fixed_exposure_analysis.{png,pdf}``: heatmaps and event summaries.

Run locally with the Homebrew Python that carries pandas/matplotlib:

    /opt/homebrew/bin/python3.11 analyze_results.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
SETUP = HERE.parents[1] / "setup" / "nm_ladder_fixed_exposure_nhop2"
MATCHED40K_H1_LADDER = HERE.parent / "nm_ladder" / "data" / "nm_ladder_full.csv"
MATCHED40K_H2_LADDER = (
    HERE.parent / "nm_ladder_nhop2" / "data" / "nm_ladder_nhop2_order_A.csv"
)

DATASETS = (
    "ukr_rus_twitter",
    "covid19_twitter",
    "midterm",
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
    "cp_hk_twitter",
)
SHORT = {
    "ukr_rus_twitter": "ukr",
    "covid19_twitter": "covid",
    "midterm": "midterm",
    "covid_political": "cov-pol",
    "election2020": "elec20",
    "ukr_rus_suspended": "ukr-susp",
    "twibot20": "twibot",
    "cp_hk_twitter": "cp-hk",
}
SOURCE_TO_DATASET = {
    "ukr_rus": "ukr_rus_twitter",
    "covid": "covid19_twitter",
    "midterm": "midterm",
    "covid_political": "covid_political",
    "election2020": "election2020",
    "ukr_rus_suspended": "ukr_rus_suspended",
    "twibot20": "twibot20",
    "cp_hk": "cp_hk_twitter",
}
ORDER_COLOR = {"A": "#2a78d6", "C": "#8f3fbf"}
ROLE_COLOR = {"newcomer": "#d85a30", "incumbent": "#2a78d6", "held_out": "#8f8d87"}
INK, MUTED, GRID = "#171715", "#6f6d68", "#e1e0d9"


def validate_raw(raw: pd.DataFrame) -> None:
    required = {
        "model", "artifact_order", "rung", "target_step", "dataset",
        "test_accuracy", "test_f1", "test_roc_auc", "evidence_path",
    }
    missing_columns = required - set(raw.columns)
    if missing_columns:
        raise ValueError(f"raw metrics missing columns: {sorted(missing_columns)}")
    if len(raw) != 120:
        raise ValueError(f"expected 120 raw evaluation rows, found {len(raw)}")
    if raw.duplicated(["model", "dataset"]).any():
        raise ValueError("raw metrics contain duplicate model/dataset cells")
    if set(raw.dataset) != set(DATASETS):
        raise ValueError(f"unexpected dataset set: {sorted(set(raw.dataset))}")
    counts = raw.groupby("model").dataset.nunique()
    if len(counts) != 15 or set(counts) != {8}:
        raise ValueError(f"expected 15 complete models, got counts={counts.to_dict()}")
    expected_steps = raw.rung.astype(int) * 10_000
    if not np.array_equal(raw.target_step.astype(int).to_numpy(), expected_steps.to_numpy()):
        raise ValueError("target_step is not rung * 10,000")
    for column in ("test_accuracy", "test_f1", "test_roc_auc"):
        if not raw[column].between(0, 1).all():
            raise ValueError(f"{column} contains values outside [0, 1]")


def load_plan(manifest_path: Path) -> pd.DataFrame:
    plan = pd.read_csv(manifest_path, sep="\t")
    plan = plan[plan.order.isin(["A", "C"])].copy()
    if len(plan) != 16 or set(zip(plan.order, plan.rung)) != {
        (order, rung) for order in ("A", "C") for rung in range(1, 9)
    }:
        raise ValueError("manifest does not contain complete A/C trajectories")
    plan["added_dataset"] = plan.added.map(SOURCE_TO_DATASET)
    if plan.added_dataset.isna().any():
        raise ValueError("manifest contains an unknown added-source key")
    plan["source_datasets"] = plan.sources.map(
        lambda value: ",".join(SOURCE_TO_DATASET[item] for item in str(value).split(","))
    )
    return plan.sort_values(["order", "rung"]).reset_index(drop=True)


def assemble_logical(raw: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    """Expand the 15 physical model matrices into 16 logical A/C rungs."""
    validate_raw(raw)
    by_model_dataset = raw.set_index(["model", "dataset"])
    if not by_model_dataset.index.is_unique:
        raise ValueError("raw model/dataset index is not unique")
    rows: list[dict[str, object]] = []
    entry = {
        (record.order, SOURCE_TO_DATASET[source]): int(record.rung)
        for record in plan.itertuples()
        for source in [record.added]
    }
    for rung in plan.itertuples():
        sources = set(str(rung.source_datasets).split(","))
        for dataset in DATASETS:
            try:
                evidence = by_model_dataset.loc[(rung.model_prefix, dataset)]
            except KeyError as exc:
                raise ValueError(
                    f"missing raw result for {rung.order} rung {rung.rung}: "
                    f"{rung.model_prefix}/{dataset}"
                ) from exc
            rows.append({
                "order": rung.order,
                "rung": int(rung.rung),
                "n_sources": int(rung.n_sources),
                "target_step": int(rung.target_step),
                "added_dataset": rung.added_dataset,
                "sources": rung.source_datasets,
                "dataset": dataset,
                "entry_rung": entry[(rung.order, dataset)],
                "rel_to_entry": int(rung.rung) - entry[(rung.order, dataset)],
                "in_training": dataset in sources,
                "test_accuracy": float(evidence.test_accuracy),
                "test_f1": float(evidence.test_f1),
                "test_roc_auc": float(evidence.test_roc_auc),
                "model": rung.model_prefix,
                "artifact_order": evidence.artifact_order,
                "shared_all8_artifact": rung.order == "C" and int(rung.rung) == 8,
                "evidence_path": evidence.evidence_path,
            })
    logical = pd.DataFrame(rows).sort_values(["order", "rung", "dataset"])
    if len(logical) != 128 or logical.duplicated(["order", "rung", "dataset"]).any():
        raise ValueError("logical A/C matrix is incomplete or duplicated")
    return logical.reset_index(drop=True)


def make_rung_summary(logical: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (order, rung), group in logical.groupby(["order", "rung"], sort=True):
        inside = group[group.in_training]
        outside = group[~group.in_training]
        rows.append({
            "order": order,
            "rung": int(rung),
            "target_step": int(group.target_step.iloc[0]),
            "n_sources": int(group.n_sources.iloc[0]),
            "added_dataset": group.added_dataset.iloc[0],
            "mean_auc": group.test_roc_auc.mean(),
            "in_training_mean_auc": inside.test_roc_auc.mean(),
            "held_out_mean_auc": outside.test_roc_auc.mean() if len(outside) else np.nan,
            "id_ood_gap": (
                inside.test_roc_auc.mean() - outside.test_roc_auc.mean()
                if len(outside) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def make_adjacent_deltas(logical: pd.DataFrame) -> pd.DataFrame:
    lookup = logical.set_index(["order", "rung", "dataset"])
    if not lookup.index.is_unique:
        raise ValueError("logical order/rung/dataset index is not unique")
    rows = []
    for order in ("A", "C"):
        for rung in range(2, 9):
            added = lookup.loc[(order, rung, DATASETS[0])].added_dataset
            previous_sources = set(lookup.loc[(order, rung - 1, DATASETS[0])].sources.split(","))
            for dataset in DATASETS:
                before = float(lookup.loc[(order, rung - 1, dataset)].test_roc_auc)
                after = float(lookup.loc[(order, rung, dataset)].test_roc_auc)
                role = (
                    "newcomer" if dataset == added
                    else "incumbent" if dataset in previous_sources
                    else "held_out"
                )
                rows.append({
                    "order": order,
                    "rung": rung,
                    "target_step": rung * 10_000,
                    "added_dataset": added,
                    "dataset": dataset,
                    "role": role,
                    "before_auc": before,
                    "after_auc": after,
                    "delta_auc": after - before,
                })
    deltas = pd.DataFrame(rows)
    if len(deltas) != 112 or (deltas.role == "newcomer").sum() != 14:
        raise ValueError("adjacent-event table has the wrong shape")
    return deltas


def two_sided_sign_p(positive: int, total: int) -> float:
    """Exact two-sided sign-test p-value, ignoring ties."""
    if total <= 0 or not 0 <= positive <= total:
        raise ValueError("invalid sign-test counts")
    k = min(positive, total - positive)
    tail = sum(math.comb(total, i) for i in range(k + 1)) / 2**total
    return min(1.0, 2 * tail)


def compare_matched40k_h1_order_a(
    logical: pd.DataFrame, matched40k_path: Path
) -> pd.DataFrame:
    """Pair Order A with the historical matched-40k one-hop ladder.

    Exposure schedule and sampler radius both differ, so this is a replication
    comparison rather than a controlled exposure ablation.
    """
    matched40k = pd.read_csv(matched40k_path)
    expected_columns = {"rung", *DATASETS}
    if not expected_columns.issubset(matched40k.columns) or len(matched40k) != 8:
        raise ValueError("matched-40k one-hop ladder table is incomplete")
    if set(matched40k.rung.astype(int)) != set(range(1, 9)):
        raise ValueError("matched-40k one-hop ladder lacks a complete Order-A trajectory")

    baseline = matched40k.set_index("rung")
    rows = []
    order_a = logical[logical.order == "A"].sort_values(["rung", "dataset"])
    for row in order_a.itertuples():
        matched_auc = float(baseline.loc[row.rung, row.dataset])
        matched_previous = (
            float(baseline.loc[row.rung - 1, row.dataset]) if row.rung > 1 else np.nan
        )
        fixed_previous = (
            float(order_a[(order_a.rung == row.rung - 1) &
                          (order_a.dataset == row.dataset)].test_roc_auc.iloc[0])
            if row.rung > 1 else np.nan
        )
        rows.append({
            "rung": row.rung,
            "dataset": row.dataset,
            "added_dataset": row.added_dataset,
            "fixed_exposure_h2_auc": row.test_roc_auc,
            "matched40k_h1_auc": matched_auc,
            "auc_difference": row.test_roc_auc - matched_auc,
            "is_entry_cell": row.rung > 1 and row.dataset == row.added_dataset,
            "fixed_entry_delta": (
                row.test_roc_auc - fixed_previous
                if row.rung > 1 and row.dataset == row.added_dataset else np.nan
            ),
            "matched40k_entry_delta": (
                matched_auc - matched_previous
                if row.rung > 1 and row.dataset == row.added_dataset else np.nan
            ),
            "comparison_scope": "cross-protocol: fixed-exposure h2 vs matched-40k h1",
        })
    return pd.DataFrame(rows)


def compare_matched40k_h2_order_a(
    logical: pd.DataFrame, matched40k_path: Path
) -> pd.DataFrame:
    """Pair Order A with the matched-40k ladder under the same fair-two-hop sampler."""
    matched40k = pd.read_csv(matched40k_path)
    expected_columns = {"rung", *DATASETS}
    if not expected_columns.issubset(matched40k.columns) or len(matched40k) != 8:
        raise ValueError("matched-40k two-hop ladder table is incomplete")
    expected_protocol = {
        "n_hop": 2,
        "hop_sizes": "9,9",
        "node_limit": 101,
        "nm_walk_hops": 1,
        "checkpoint_step": 40_000,
        "order": "A",
    }
    for column, expected in expected_protocol.items():
        if column not in matched40k.columns:
            raise ValueError(f"matched-40k table lacks protocol column {column!r}")
        values = set(matched40k[column].tolist())
        if values != {expected}:
            raise ValueError(
                f"matched-40k table has {column}={values}, expected {expected!r}"
            )
    if set(matched40k.rung.astype(int)) != set(range(1, 9)):
        raise ValueError("matched-40k two-hop ladder lacks a complete Order-A trajectory")

    baseline = matched40k.set_index("rung")
    rows = []
    order_a = logical[logical.order == "A"].sort_values(["rung", "dataset"])
    for row in order_a.itertuples():
        matched_auc = float(baseline.loc[row.rung, row.dataset])
        matched_previous = (
            float(baseline.loc[row.rung - 1, row.dataset]) if row.rung > 1 else np.nan
        )
        fixed_previous = (
            float(order_a[(order_a.rung == row.rung - 1) &
                          (order_a.dataset == row.dataset)].test_roc_auc.iloc[0])
            if row.rung > 1 else np.nan
        )
        rows.append({
            "rung": row.rung,
            "dataset": row.dataset,
            "added_dataset": row.added_dataset,
            "entry_rung": row.entry_rung,
            "fixed_total_steps": row.target_step,
            "matched40k_total_steps": 40_000,
            "fixed_expected_steps_per_source": 10_000,
            "matched40k_expected_steps_per_source": 40_000 / row.rung,
            "fixed_exposure_h2_auc": row.test_roc_auc,
            "matched40k_h2_auc": matched_auc,
            "auc_difference": row.test_roc_auc - matched_auc,
            "is_entry_cell": row.rung > 1 and row.dataset == row.added_dataset,
            "fixed_entry_delta": (
                row.test_roc_auc - fixed_previous
                if row.rung > 1 and row.dataset == row.added_dataset else np.nan
            ),
            "matched40k_h2_entry_delta": (
                matched_auc - matched_previous
                if row.rung > 1 and row.dataset == row.added_dataset else np.nan
            ),
            "comparison_scope": (
                "controlled Order A: same fair-two-hop sampler and source sets; "
                "fixed 10k/source vs fixed 40k total"
            ),
        })
    return pd.DataFrame(rows)


def headline_summary(
    logical: pd.DataFrame,
    rung_summary: pd.DataFrame,
    deltas: pd.DataFrame,
    matched40k_h1: pd.DataFrame | None = None,
    matched40k_h2: pd.DataFrame | None = None,
) -> dict[str, object]:
    entry = deltas[deltas.role == "newcomer"]
    incumbent = deltas[deltas.role == "incumbent"]
    held_out = deltas[deltas.role == "held_out"]
    retention = []
    for (order, dataset), group in logical.groupby(["order", "dataset"]):
        entry_rung = int(group.entry_rung.iloc[0])
        if entry_rung == 8:
            continue
        entry_auc = float(group[group.rung == entry_rung].test_roc_auc.iloc[0])
        final_auc = float(group[group.rung == 8].test_roc_auc.iloc[0])
        retention.append(final_auc - entry_auc)
    positive = int((entry.delta_auc > 0).sum())
    result: dict[str, object] = {
        "physical_metric_rows": 120,
        "logical_metric_rows": 128,
        "orders": ["A", "C"],
        "entry_events": int(len(entry)),
        "positive_entry_events": positive,
        "entry_jump_mean": float(entry.delta_auc.mean()),
        "entry_jump_median": float(entry.delta_auc.median()),
        "entry_jump_min": float(entry.delta_auc.min()),
        "entry_jump_max": float(entry.delta_auc.max()),
        "entry_jump_sign_test_two_sided_p": two_sided_sign_p(positive, len(entry)),
        "incumbent_delta_mean": float(incumbent.delta_auc.mean()),
        "incumbent_delta_median": float(incumbent.delta_auc.median()),
        "incumbent_negative_fraction": float((incumbent.delta_auc < 0).mean()),
        "held_out_delta_mean": float(held_out.delta_auc.mean()),
        "held_out_delta_median": float(held_out.delta_auc.median()),
        "post_entry_to_all8_delta_mean": float(np.mean(retention)),
        "post_entry_to_all8_delta_median": float(np.median(retention)),
        "order_mean_auc": {
            order: {
                str(int(row.rung)): float(row.mean_auc)
                for row in rung_summary[rung_summary.order == order].itertuples()
            }
            for order in ("A", "C")
        },
        "all8_mean_auc": float(
            rung_summary[(rung_summary.order == "A") & (rung_summary.rung == 8)].mean_auc.iloc[0]
        ),
    }
    if matched40k_h1 is not None:
        entry_comparison = matched40k_h1[matched40k_h1.is_entry_cell].copy()
        entry_comparison["entry_delta_difference"] = (
            entry_comparison.fixed_entry_delta
            - entry_comparison.matched40k_entry_delta
        )
        result["historical_matched40k_h1_comparison"] = {
            "scope": "cross-protocol; exposure and sampler radius both differ",
            "paired_cells": int(len(matched40k_h1)),
            "mean_auc_difference": float(matched40k_h1.auc_difference.mean()),
            "mean_absolute_auc_difference": float(
                matched40k_h1.auc_difference.abs().mean()
            ),
            "entry_events": int(len(entry_comparison)),
            "entry_delta_mean_difference": float(
                entry_comparison.entry_delta_difference.mean()
            ),
            "entry_delta_mean_absolute_difference": float(
                entry_comparison.entry_delta_difference.abs().mean()
            ),
        }
    if matched40k_h2 is not None:
        entry_comparison = matched40k_h2[matched40k_h2.is_entry_cell].copy()
        entry_comparison["entry_delta_difference"] = (
            entry_comparison.fixed_entry_delta
            - entry_comparison.matched40k_h2_entry_delta
        )
        retention_rows = []
        for dataset, group in matched40k_h2.groupby("dataset"):
            entry_rung = int(group.entry_rung.iloc[0])
            if entry_rung == 8:
                continue
            by_rung = group.set_index("rung")
            retention_rows.append({
                "dataset": dataset,
                "fixed": (
                    by_rung.loc[8, "fixed_exposure_h2_auc"]
                    - by_rung.loc[entry_rung, "fixed_exposure_h2_auc"]
                ),
                "matched40k": (
                    by_rung.loc[8, "matched40k_h2_auc"]
                    - by_rung.loc[entry_rung, "matched40k_h2_auc"]
                ),
            })
        retention_comparison = pd.DataFrame(retention_rows)
        retention_comparison["difference"] = (
            retention_comparison.fixed - retention_comparison.matched40k
        )

        rung_means = matched40k_h2.groupby("rung")[[
            "fixed_exposure_h2_auc", "matched40k_h2_auc"
        ]].mean()
        result["matched40k_h2_order_a_comparison"] = {
            "scope": (
                "controlled Order A; same fair-two-hop sampler, source sets, and eval; "
                "one training seed"
            ),
            "paired_cells": int(len(matched40k_h2)),
            "mean_auc_difference": float(matched40k_h2.auc_difference.mean()),
            "median_auc_difference": float(matched40k_h2.auc_difference.median()),
            "mean_absolute_auc_difference": float(
                matched40k_h2.auc_difference.abs().mean()
            ),
            "min_auc_difference": float(matched40k_h2.auc_difference.min()),
            "max_auc_difference": float(matched40k_h2.auc_difference.max()),
            "rung1_fixed_exposure_mean_auc": float(
                rung_means.loc[1, "fixed_exposure_h2_auc"]
            ),
            "rung1_matched40k_mean_auc": float(rung_means.loc[1, "matched40k_h2_auc"]),
            "rung1_mean_auc_difference": float(
                rung_means.loc[1, "fixed_exposure_h2_auc"]
                - rung_means.loc[1, "matched40k_h2_auc"]
            ),
            "rung4_mean_auc_difference": float(
                rung_means.loc[4, "fixed_exposure_h2_auc"]
                - rung_means.loc[4, "matched40k_h2_auc"]
            ),
            "rung8_mean_auc_difference": float(
                rung_means.loc[8, "fixed_exposure_h2_auc"]
                - rung_means.loc[8, "matched40k_h2_auc"]
            ),
            "entry_events": int(len(entry_comparison)),
            "fixed_exposure_entry_jump_mean": float(
                entry_comparison.fixed_entry_delta.mean()
            ),
            "matched40k_entry_jump_mean": float(
                entry_comparison.matched40k_h2_entry_delta.mean()
            ),
            "entry_delta_mean_difference": float(entry_comparison.entry_delta_difference.mean()),
            "entry_delta_mean_absolute_difference": float(
                entry_comparison.entry_delta_difference.abs().mean()
            ),
            "fixed_exposure_post_entry_retention_mean": float(
                retention_comparison.fixed.mean()
            ),
            "matched40k_post_entry_retention_mean": float(
                retention_comparison.matched40k.mean()
            ),
            "post_entry_retention_mean_difference": float(
                retention_comparison.difference.mean()
            ),
        }
    return result


def _chrome(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.set_axisbelow(True)


def make_figure(logical: pd.DataFrame, deltas: pd.DataFrame, out_dir: Path) -> None:
    fig = plt.figure(figsize=(15, 10), dpi=180)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.12, 0.88), hspace=0.35, wspace=0.22)
    heat_axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]

    im = None
    for ax, order in zip(heat_axes, ("A", "C")):
        sub = logical[logical.order == order]
        matrix = (
            sub.pivot(index="rung", columns="dataset", values="test_roc_auc")
            .reindex(index=range(1, 9), columns=DATASETS)
        )
        im = ax.imshow(matrix, cmap="viridis", vmin=0.58, vmax=0.99, aspect="auto")
        for yi, rung in enumerate(range(1, 9)):
            added = sub[sub.rung == rung].added_dataset.iloc[0]
            for xi, dataset in enumerate(DATASETS):
                value = float(matrix.loc[rung, dataset])
                color = "white" if value < 0.79 else INK
                ax.text(xi, yi, f"{value:.3f}", ha="center", va="center",
                        fontsize=7.4, color=color)
                if dataset == added:
                    ax.add_patch(Rectangle((xi - 0.47, yi - 0.47), 0.94, 0.94,
                                           fill=False, edgecolor="#ff9c54", linewidth=2.0))
        ax.set_xticks(range(8), [SHORT[d] for d in DATASETS], rotation=35, ha="right")
        ax.set_yticks(range(8), [f"r{r} · {r*10}k" for r in range(1, 9)])
        ax.set_xlabel("evaluation graph", color=MUTED)
        ax.set_ylabel("mixture rung · total steps", color=MUTED)
        ax.set_title(f"Order {order}", loc="left", fontsize=12, fontweight="bold", color=INK)
        ax.tick_params(labelsize=8.3, colors=MUTED)
    assert im is not None
    cbar = fig.colorbar(im, ax=heat_axes, shrink=0.82, pad=0.02)
    cbar.set_label("NM test ROC-AUC", color=MUTED)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)

    # Entry-aligned trajectories. Each line is one graph/order pair; the mean is
    # unbalanced at extreme offsets, so coverage is written directly under the axis.
    ax = fig.add_subplot(grid[1, 0])
    at_rel: dict[int, list[float]] = {}
    for (_, _), group in logical.groupby(["order", "dataset"]):
        series = group.sort_values("rel_to_entry")
        ax.plot(series.rel_to_entry, series.test_roc_auc, color="#aaa8a2",
                linewidth=0.75, alpha=0.34, zorder=1)
        for row in series.itertuples():
            at_rel.setdefault(int(row.rel_to_entry), []).append(float(row.test_roc_auc))
    keep = sorted(rel for rel, values in at_rel.items() if len(values) >= 4)
    means = [np.mean(at_rel[rel]) for rel in keep]
    lows = [np.min(at_rel[rel]) for rel in keep]
    highs = [np.max(at_rel[rel]) for rel in keep]
    ax.fill_between(keep, lows, highs, color="#2a78d6", alpha=0.10, linewidth=0)
    ax.plot(keep, means, color=INK, linewidth=2.5, marker="o", markersize=5.5,
            markeredgecolor="white", markeredgewidth=1.0, zorder=3)
    ax.axvline(0, color="#d85a30", linestyle=(0, (3, 2)), linewidth=1.5)
    ax.text(0.12, 0.60, "enters mixture", color="#d85a30", fontsize=9, fontweight="bold")
    entry = deltas[deltas.role == "newcomer"]
    ax.annotate(f"14/14 jumps positive\nmean {entry.delta_auc.mean():+.3f}",
                xy=(0, np.mean(at_rel[0])), xytext=(1.25, 0.67),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1),
                color=INK, fontsize=9, fontweight="bold")
    ax.set_xlabel("rungs relative to graph entry (0 = enters)", color=MUTED)
    ax.set_ylabel("NM test ROC-AUC", color=MUTED)
    ax.set_title("Entry-aligned trajectories", loc="left", fontsize=11.5,
                 fontweight="bold", color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    _chrome(ax)

    # Every adjacent cell delta by its role during the source-addition event.
    ax = fig.add_subplot(grid[1, 1])
    roles = ("newcomer", "incumbent", "held_out")
    labels = ("newcomer\n(enters now)", "incumbent\n(already in)", "held-out\n(not yet in)")
    rng = np.random.default_rng(0)
    values = [deltas[deltas.role == role].delta_auc.to_numpy() for role in roles]
    boxes = ax.boxplot(values, positions=range(3), widths=0.48, patch_artist=True,
                       showfliers=False, medianprops=dict(color=INK, linewidth=1.5),
                       whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    for box, role in zip(boxes["boxes"], roles):
        box.set(facecolor=ROLE_COLOR[role], alpha=0.18, edgecolor=ROLE_COLOR[role])
    for x, (role, vals) in enumerate(zip(roles, values)):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), x) + jitter, vals, s=20,
                   color=ROLE_COLOR[role], alpha=0.62, edgecolor="white", linewidth=0.4)
        ax.text(x, 0.375, f"mean {np.mean(vals):+.3f}\nn={len(vals)}",
                ha="center", va="bottom", fontsize=8.3, color=INK)
    ax.axhline(0, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("adjacent-rung Δ ROC-AUC", color=MUTED)
    ax.set_ylim(-0.065, 0.42)
    ax.set_title("What changes when one source is added?", loc="left", fontsize=11.5,
                 fontweight="bold", color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    _chrome(ax)

    fig.suptitle("Fixed exposure preserves the NM interpolation staircase",
                 x=0.07, ha="left", y=0.99, fontsize=15, fontweight="bold", color=INK)
    fig.text(0.07, 0.955,
             "Fair two-hop sampler · 10k expected episodes per active source · "
             "orange boxes mark the source added at each rung · one seed",
             ha="left", fontsize=9.5, color=MUTED)
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = out_dir / f"fixed_exposure_analysis.{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=HERE / "data" / "raw_metrics.csv")
    parser.add_argument("--manifest", type=Path, default=SETUP / "manifest.tsv")
    parser.add_argument(
        "--matched40k-h1-ladder", type=Path, default=MATCHED40K_H1_LADDER
    )
    parser.add_argument(
        "--matched40k-h2-ladder", type=Path, default=MATCHED40K_H2_LADDER
    )
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument(
        "--skip-figures", action="store_true",
        help="regenerate tables and JSON without rewriting unchanged figures",
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.raw)
    plan = load_plan(args.manifest)
    logical = assemble_logical(raw, plan)
    rung_summary = make_rung_summary(logical)
    deltas = make_adjacent_deltas(logical)
    entry = deltas[deltas.role == "newcomer"].copy()
    matched40k_h1 = compare_matched40k_h1_order_a(
        logical, args.matched40k_h1_ladder
    )
    matched40k_h2 = compare_matched40k_h2_order_a(
        logical, args.matched40k_h2_ladder
    )
    summary = headline_summary(
        logical, rung_summary, deltas, matched40k_h1, matched40k_h2
    )

    data_dir = args.out_dir / "data"
    figure_dir = args.out_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    logical.to_csv(data_dir / "logical_results.csv", index=False)
    rung_summary.to_csv(data_dir / "rung_summary.csv", index=False)
    deltas.to_csv(data_dir / "adjacent_deltas.csv", index=False)
    entry.to_csv(data_dir / "entry_jumps.csv", index=False)
    matched40k_h1.to_csv(
        data_dir / "comparison_to_matched40k_h1_orderA.csv", index=False
    )
    matched40k_h2.to_csv(
        data_dir / "comparison_to_matched40k_h2_orderA.csv", index=False
    )
    (data_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not args.skip_figures:
        make_figure(logical, deltas, figure_dir)

    print("\nRung means")
    print(rung_summary.pivot(index="rung", columns="order", values="mean_auc").round(4))
    print("\nEntry jumps")
    print(entry[["order", "rung", "added_dataset", "before_auc", "after_auc", "delta_auc"]]
          .to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nHeadline summary")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
