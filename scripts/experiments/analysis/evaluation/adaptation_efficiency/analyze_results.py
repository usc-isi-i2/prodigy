#!/usr/bin/env python3
"""Summarize and plot the frozen-head adaptation-efficiency grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.experiments.setup.adaptation_efficiency.protocol import UPDATE_STEPS


ORDER = ["PRODIGY", "VISION", "SAMGPT", "GraphSAGE", "Raw logistic", "Raw MLP"]
COLORS = dict(zip(ORDER, plt.get_cmap("tab10").colors[: len(ORDER)]))
EXPECTED_MODELS = {
    *(f"prodigy_all9_s{seed}" for seed in (0, 1, 2)),
    *(f"vision_all9_s{seed}" for seed in (0, 1, 2)),
    *(f"samgpt_all9_s{seed}" for seed in (39, 40, 41)),
    "graphsage_pilot_v1",
    "raw_logistic",
    "raw_mlp",
}
EXPECTED_TARGETS = {
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
}
CELLS_PER_SEED = 1 + 3 * len(UPDATE_STEPS)
EXPECTED_ROWS = len(EXPECTED_MODELS) * len(EXPECTED_TARGETS) * 3 * CELLS_PER_SEED * 2


def family(model_id: str) -> str:
    if model_id.startswith("prodigy_"):
        return "PRODIGY"
    if model_id.startswith("vision_"):
        return "VISION"
    if model_id.startswith("samgpt_"):
        return "SAMGPT"
    if model_id.startswith("graphsage_"):
        return "GraphSAGE"
    if model_id == "raw_logistic":
        return "Raw logistic"
    if model_id == "raw_mlp":
        return "Raw MLP"
    return model_id


def save_figure(fig, root: Path, name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def validate_shared_protocol(cells: pd.DataFrame) -> None:
    """Reject results that violate the promised shared sampling/head contract."""
    required_columns = {
        "model_id", "target", "label_seed", "label_budget_per_class", "head_updates",
        "split", "selected_nodes_fingerprint", "split_fingerprint",
        "head_initialization_fingerprint", "optimizer", "learning_rate", "weight_decay",
        "roc_auc", "accuracy", "macro_f1", "training_loss", "training_roc_auc",
        "training_accuracy", "training_macro_f1",
    }
    if missing := required_columns - set(cells):
        raise ValueError(f"adaptation cells lack columns: {sorted(missing)}")
    if len(cells) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} adaptation rows, got {len(cells)}")
    if set(cells.model_id) != EXPECTED_MODELS:
        raise ValueError("adaptation model registry mismatch")
    if set(cells.target) != EXPECTED_TARGETS or set(cells.label_seed) != {0, 1, 2}:
        raise ValueError("adaptation target or label-seed registry mismatch")
    if set(cells.split) != {"val", "test"}:
        raise ValueError("adaptation output must contain validation and test rows only")
    keys = [
        "model_id", "target", "label_seed", "label_budget_per_class", "head_updates", "split"
    ]
    if cells.duplicated(keys).any():
        raise ValueError("duplicate adaptation cells")
    test = cells[cells.split == "test"].copy()
    split_counts = test.groupby("target")["split_fingerprint"].nunique()
    if not split_counts.eq(1).all():
        raise ValueError(f"split fingerprints differ across models: {split_counts.to_dict()}")
    sample_counts = test.groupby(
        ["target", "label_seed", "label_budget_per_class"]
    )["selected_nodes_fingerprint"].nunique()
    if not sample_counts.eq(1).all():
        raise ValueError("labeled-node samples differ across model families")
    linear = test[test.model_id != "raw_mlp"]
    initialization_counts = linear.groupby(
        ["target", "label_seed"]
    )["head_initialization_fingerprint"].nunique()
    if not initialization_counts.eq(1).all():
        raise ValueError("learned encoders and raw logistic do not share identical linear heads")
    positive = test[test.label_budget_per_class > 0]
    if (
        set(positive.optimizer) != {"AdamW"}
        or set(positive.learning_rate) != {0.01}
        or set(positive.weight_decay) != {0.0}
    ):
        raise ValueError("positive-label cells do not share the registered optimizer")
    zero = test[test.label_budget_per_class == 0]
    if (
        set(zero.head_updates) != {0}
        or set(zero.optimizer) != {"none"}
        or set(zero.learning_rate) != {0.0}
    ):
        raise ValueError("zero-label cells contain optimizer updates")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data_dir = args.output / "data"
    figure_dir = args.output / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(args.cells)
    cells["family"] = cells.model_id.map(family)
    validate_shared_protocol(cells)
    cells.to_csv(data_dir / "adaptation_cells_full.csv", index=False)
    test = cells[cells.split == "test"].copy()

    coverage = (
        test.groupby(["family", "model_id", "target"], as_index=False)
        .agg(rows=("roc_auc", "size"), label_seeds=("label_seed", "nunique"))
    )
    coverage["expected_rows"] = 3 * CELLS_PER_SEED
    coverage["complete"] = coverage.rows == coverage.expected_rows
    coverage.to_csv(data_dir / "coverage.csv", index=False)
    if not coverage.complete.all():
        incomplete = coverage.loc[~coverage.complete, ["model_id", "target", "rows"]]
        raise ValueError(f"incomplete model-target grids: {incomplete.to_dict('records')}")

    curve = (
        test.groupby(["family", "label_budget_per_class", "head_updates"], as_index=False)
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            cells=("roc_auc", "size"),
        )
    )
    curve.to_csv(data_dir / "learning_curves.csv", index=False)
    budgets = [0, 1, 10, 100]
    update_milestones = list(UPDATE_STEPS)
    update_positions = {update: position for position, update in enumerate(update_milestones)}
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    for axis, budget in zip(axes, budgets):
        panel = curve[curve.label_budget_per_class == budget]
        for name in ORDER:
            rows = panel[panel.family == name].sort_values("head_updates")
            if rows.empty:
                continue
            # Updates are four registered evaluation milestones, not a continuous
            # trajectory. Plot them at discrete positions so update 0 remains visible
            # without the misleading negative tick produced by a symlog axis.
            x = rows.head_updates.map(update_positions)
            axis.plot(x, rows.roc_auc_mean, marker="o", label=name, color=COLORS[name])
        title = "Untrained head\n(0 labels/class)" if budget == 0 else (
            f"{budget} label{'s' if budget != 1 else ''}/class"
        )
        axis.set_title(title)
        axis.set_xlabel("Head updates")
        if budget == 0:
            axis.set_xticks([update_positions[0]], ["0"])
            axis.set_xlim(-0.5, 0.5)
        else:
            axis.set_xticks(range(len(update_milestones)), [str(x) for x in update_milestones])
            axis.set_xlim(-0.15, len(update_milestones) - 0.85)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Test ROC-AUC")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=6, frameon=False,
    )
    fig.suptitle("Frozen-encoder optimization efficiency", y=0.90)
    fig.subplots_adjust(top=0.76)
    save_figure(fig, figure_dir, "optimization_learning_curves")

    training_curve = (
        test[test.label_budget_per_class > 0]
        .groupby(["family", "label_budget_per_class", "head_updates"], as_index=False)
        .agg(
            training_loss_mean=("training_loss", "mean"),
            training_loss_std=("training_loss", "std"),
            training_roc_auc_mean=("training_roc_auc", "mean"),
        )
    )
    training_curve.to_csv(data_dir / "training_curves.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, budget in zip(axes, (1, 10, 100)):
        panel = training_curve[training_curve.label_budget_per_class == budget]
        for name in ORDER:
            rows = panel[panel.family == name].sort_values("head_updates")
            if rows.empty:
                continue
            x = rows.head_updates.map(update_positions)
            axis.plot(x, rows.training_loss_mean, marker="o", label=name, color=COLORS[name])
        axis.set_title(f"{budget} label{'s' if budget != 1 else ''}/class")
        axis.set_xlabel("Head updates")
        axis.set_xticks(range(len(update_milestones)), [str(x) for x in update_milestones])
        axis.set_xlim(-0.15, len(update_milestones) - 0.85)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Labeled-train cross-entropy")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=6, frameon=False)
    fig.suptitle("Frozen-encoder head fit", y=0.90)
    fig.subplots_adjust(top=0.76)
    save_figure(fig, figure_dir, "training_loss_curves")

    selection_keys = ["model_id", "target", "label_seed", "label_budget_per_class"]
    validation_selected = (
        cells[(cells.split == "val") & (cells.label_budget_per_class > 0)]
        .sort_values(selection_keys + ["roc_auc", "head_updates"], ascending=[True] * 4 + [False, True])
        .drop_duplicates(selection_keys, keep="first")
        [selection_keys + ["head_updates", "roc_auc"]]
        .rename(columns={"head_updates": "selected_head_updates", "roc_auc": "validation_roc_auc"})
    )
    selected_test = test[test.label_budget_per_class > 0].merge(
        validation_selected,
        left_on=selection_keys + ["head_updates"],
        right_on=selection_keys + ["selected_head_updates"],
        how="inner",
        validate="one_to_one",
    )
    if len(selected_test) != len(EXPECTED_MODELS) * len(EXPECTED_TARGETS) * 3 * 3:
        raise ValueError("validation-selected test grid is incomplete")
    selected_test.to_csv(data_dir / "validation_selected_test.csv", index=False)
    selected_summary = (
        selected_test.groupby(["family", "label_budget_per_class"], as_index=False)
        .agg(
            test_roc_auc_mean=("roc_auc", "mean"),
            test_roc_auc_std=("roc_auc", "std"),
            median_selected_updates=("selected_head_updates", "median"),
            cells=("roc_auc", "size"),
        )
    )
    selected_summary.to_csv(data_dir / "validation_selected_summary.csv", index=False)

    late = (
        test[(test.label_budget_per_class > 0) & test.head_updates.isin([10, 100])]
        .groupby(["family", "label_budget_per_class", "head_updates"], as_index=False)
        .agg(training_loss=("training_loss", "mean"), test_roc_auc=("roc_auc", "mean"))
    )
    late_10 = late[late.head_updates == 10].drop(columns="head_updates").rename(
        columns={"training_loss": "training_loss_10", "test_roc_auc": "test_roc_auc_10"}
    )
    late_100 = late[late.head_updates == 100].drop(columns="head_updates").rename(
        columns={"training_loss": "training_loss_100", "test_roc_auc": "test_roc_auc_100"}
    )
    late_change = late_10.merge(
        late_100, on=["family", "label_budget_per_class"], validate="one_to_one"
    )
    late_change["training_loss_delta_100_minus_10"] = (
        late_change.training_loss_100 - late_change.training_loss_10
    )
    late_change["test_roc_auc_delta_100_minus_10"] = (
        late_change.test_roc_auc_100 - late_change.test_roc_auc_10
    )
    late_change = late_change.merge(
        selected_summary[["family", "label_budget_per_class", "test_roc_auc_mean"]],
        on=["family", "label_budget_per_class"],
        validate="one_to_one",
    ).rename(columns={"test_roc_auc_mean": "validation_selected_test_roc_auc"})
    late_change["validation_selection_gain_over_update_100"] = (
        late_change.validation_selected_test_roc_auc - late_change.test_roc_auc_100
    )
    late_change.to_csv(data_dir / "late_training_diagnostics.csv", index=False)

    endpoint = test[
        ((test.label_budget_per_class == 0) & (test.head_updates == 0))
        | ((test.label_budget_per_class > 0) & (test.head_updates == 100))
    ]
    label_curve = (
        endpoint.groupby(["family", "label_budget_per_class"], as_index=False)
        .agg(roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"))
    )
    label_curve["log10_budget_plus_one"] = np.log10(label_curve.label_budget_per_class + 1)
    label_curve.to_csv(data_dir / "label_efficiency.csv", index=False)
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    for name in ORDER:
        rows = label_curve[label_curve.family == name].sort_values("label_budget_per_class")
        if rows.empty:
            continue
        axis.plot(
            rows.log10_budget_plus_one,
            rows.roc_auc_mean,
            marker="o",
            label=name,
            color=COLORS[name],
        )
    axis.set_xticks(np.log10(np.asarray(budgets) + 1), budgets)
    axis.set_xlabel("Labeled examples per class (log scale)")
    axis.set_ylabel("Test ROC-AUC at final scheduled update")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, frameon=False)
    save_figure(fig, figure_dir, "label_efficiency")

    auc_rows = []
    for keys, rows in endpoint.groupby(["family", "model_id", "target", "label_seed"]):
        rows = rows.sort_values("label_budget_per_class")
        if rows.label_budget_per_class.tolist() != budgets:
            continue
        x = np.log10(rows.label_budget_per_class.to_numpy() + 1)
        auc_rows.append(
            dict(
                zip(("family", "model_id", "target", "label_seed"), keys),
                label_efficiency_auc=float(np.trapz(rows.roc_auc, x) / (x[-1] - x[0])),
            )
        )
    auc = pd.DataFrame(auc_rows)
    auc.to_csv(data_dir / "label_efficiency_auc.csv", index=False)

    reaching = []
    positive = test[test.label_budget_per_class > 0]
    group_cols = ["family", "model_id", "target", "label_seed", "label_budget_per_class"]
    for keys, rows in positive.groupby(group_cols):
        rows = rows.sort_values("head_updates")
        final = float(rows.loc[rows.head_updates == 100, "roc_auc"].iloc[0])
        eligible = rows[rows.roc_auc >= 0.95 * final]
        reaching.append(
            dict(
                zip(group_cols, keys),
                final_roc_auc=final,
                updates_to_95pct=int(eligible.head_updates.min()),
            )
        )
    reaching_frame = pd.DataFrame(reaching)
    reaching_frame.to_csv(data_dir / "updates_to_95pct.csv", index=False)
    summary_updates = (
        reaching_frame.groupby(["family", "label_budget_per_class"], as_index=False)
        .agg(median_updates=("updates_to_95pct", "median"), mean_updates=("updates_to_95pct", "mean"))
    )
    summary_updates.to_csv(data_dir / "updates_to_95pct_summary.csv", index=False)
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    width = 0.12
    x = np.arange(3)
    for offset, name in enumerate(ORDER):
        rows = summary_updates[summary_updates.family == name].set_index("label_budget_per_class")
        if rows.empty:
            continue
        values = [rows.loc[budget, "median_updates"] for budget in (1, 10, 100)]
        axis.bar(x + (offset - 2.5) * width, values, width, label=name, color=COLORS[name])
    axis.set_xticks(x, ["1", "10", "100"])
    axis.set_xlabel("Labeled examples per class")
    axis.set_ylabel("Median updates to 95% of update-100 ROC-AUC")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2, frameon=False)
    save_figure(fig, figure_dir, "updates_to_95pct")

    family_summary = (
        auc.groupby("family", as_index=False)
        .agg(
            label_efficiency_auc_mean=("label_efficiency_auc", "mean"),
            label_efficiency_auc_std=("label_efficiency_auc", "std"),
            cells=("label_efficiency_auc", "size"),
        )
        .sort_values("label_efficiency_auc_mean", ascending=False)
    )
    family_summary.to_csv(data_dir / "family_summary.csv", index=False)
    manifest = {
        "source": str(args.cells),
        "rows": int(len(cells)),
        "test_rows": int(len(test)),
        "families": sorted(test.family.unique()),
        "models": int(test.model_id.nunique()),
        "targets": sorted(test.target.unique()),
        "label_seeds": sorted(int(value) for value in test.label_seed.unique()),
        "complete_model_target_grids": int(coverage.complete.sum()),
        "expected_model_target_grids": int(len(coverage)),
    }
    (data_dir / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")

    lines = [
        "# Adaptation-efficiency results",
        "",
        f"Observed {len(cells):,} validation/test rows across {test.model_id.nunique()} model checkpoints "
        f"and {test.target.nunique()} targets. Complete model-target grids: "
        f"{int(coverage.complete.sum())}/{len(coverage)}.",
        "",
        "## Label-efficiency summary",
        "",
        "| Family | mean normalized AUC over log10(labels + 1) | SD | curves |",
        "|---|---:|---:|---:|",
    ]
    for row in family_summary.itertuples():
        lines.append(
            f"| {row.family} | {row.label_efficiency_auc_mean:.4f} | "
            f"{row.label_efficiency_auc_std:.4f} | {row.cells} |"
        )
    falling_loss = int((late_change.training_loss_delta_100_minus_10 < 0).sum())
    falling_test = int((late_change.test_roc_auc_delta_100_minus_10 < 0).sum())
    lines += [
        "",
        "## Late-training diagnostic",
        "",
        f"From update 10 to 100, labeled-training loss fell in {falling_loss}/{len(late_change)} "
        f"family-by-budget curves while test ROC-AUC fell in {falling_test}/{len(late_change)}. "
        "Falling training loss alongside falling validation/test performance is evidence of "
        "head overfitting, not optimizer divergence or encoder drift.",
        "",
        "| Family | labels/class | Δ train loss (100−10) | Δ test AUC (100−10) | "
        "validation-selection gain vs 100 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in late_change.itertuples():
        lines.append(
            f"| {row.family} | {row.label_budget_per_class} | "
            f"{row.training_loss_delta_100_minus_10:+.4f} | "
            f"{row.test_roc_auc_delta_100_minus_10:+.4f} | "
            f"{row.validation_selection_gain_over_update_100:+.4f} |"
        )
    lines += [
        "",
        "## Optimization-efficiency summary",
        "",
        "Median head updates required to reach 95% of each curve's update-100 ROC-AUC:",
        "",
        "| Family | 1 label/class | 10 labels/class | 100 labels/class |",
        "|---|---:|---:|---:|",
    ]
    for name in ORDER:
        rows = summary_updates[summary_updates.family == name].set_index("label_budget_per_class")
        lines.append(
            f"| {name} | {rows.loc[1, 'median_updates']:.1f} | "
            f"{rows.loc[10, 'median_updates']:.1f} | {rows.loc[100, 'median_updates']:.1f} |"
        )
    lines += [
        "",
        "All summaries retain every label-seed, target, training-seed, label-budget, and update cell. "
        "The zero-label point is an untrained-head baseline and has no optimizer updates.",
        "",
    ]
    (args.output / "FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
