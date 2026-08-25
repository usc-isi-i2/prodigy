#!/usr/bin/env python3
"""Validate and plot GraphSAGE saturation under the shared frozen-head protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
STEPS = (0, 20, 60, 100, 300, 900, 2000)
TARGETS = ("covid_political", "election2020", "ukr_rus_suspended", "twibot20")
LABEL_SEEDS = (0, 1, 2)
BUDGETS = (0, 1, 10, 100)
UPDATES = (0, 1, 10, 100)
EXPECTED_MODELS = {f"graphsage_pilot_v1_step{step}" for step in STEPS}
EXPECTED_ROWS = len(STEPS) * len(TARGETS) * len(LABEL_SEEDS) * 13 * 2


def pretraining_step(model_id: str) -> int:
    prefix = "graphsage_pilot_v1_step"
    if not model_id.startswith(prefix):
        raise ValueError(f"unexpected GraphSAGE trajectory model id: {model_id}")
    return int(model_id.removeprefix(prefix))


def validate_cells(cells: pd.DataFrame) -> None:
    required = {
        "model_id", "target", "label_seed", "label_budget_per_class", "head_updates",
        "split", "selected_nodes_fingerprint", "split_fingerprint",
        "head_initialization_fingerprint", "optimizer", "learning_rate", "weight_decay",
        "roc_auc", "accuracy", "macro_f1",
    }
    if missing := required - set(cells):
        raise ValueError(f"GraphSAGE saturation cells lack columns: {sorted(missing)}")
    if len(cells) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} GraphSAGE saturation rows, got {len(cells)}")
    if set(cells.model_id) != EXPECTED_MODELS:
        raise ValueError("GraphSAGE trajectory checkpoint registry mismatch")
    if set(cells.target) != set(TARGETS) or set(cells.label_seed) != set(LABEL_SEEDS):
        raise ValueError("GraphSAGE trajectory target or label-seed registry mismatch")
    if set(cells.split) != {"val", "test"}:
        raise ValueError("GraphSAGE trajectory must retain validation and test rows")
    keys = [
        "model_id", "target", "label_seed", "label_budget_per_class", "head_updates", "split"
    ]
    if cells.duplicated(keys).any():
        raise ValueError("duplicate GraphSAGE saturation cells")

    test = cells[cells.split == "test"]
    coverage = test.groupby(["model_id", "target"]).size()
    if len(coverage) != len(STEPS) * len(TARGETS) or not coverage.eq(39).all():
        raise ValueError("incomplete GraphSAGE checkpoint-target head grids")
    if not test.groupby("target")["split_fingerprint"].nunique().eq(1).all():
        raise ValueError("GraphSAGE trajectory split fingerprints changed across checkpoints")
    samples = test.groupby(
        ["target", "label_seed", "label_budget_per_class"]
    )["selected_nodes_fingerprint"].nunique()
    if not samples.eq(1).all():
        raise ValueError("GraphSAGE trajectory labeled-node samples changed across checkpoints")
    initializations = test.groupby(
        ["target", "label_seed"]
    )["head_initialization_fingerprint"].nunique()
    if not initializations.eq(1).all():
        raise ValueError("GraphSAGE trajectory head initialization changed across checkpoints")
    zero = test[test.label_budget_per_class == 0]
    if set(zero.head_updates) != {0} or set(zero.optimizer) != {"none"}:
        raise ValueError("zero-label GraphSAGE cells contain optimizer updates")
    positive = test[test.label_budget_per_class > 0]
    if (
        set(positive.optimizer) != {"AdamW"}
        or set(positive.learning_rate) != {0.01}
        or set(positive.weight_decay) != {0.0}
    ):
        raise ValueError("positive-label GraphSAGE cells changed optimizer protocol")


def save_figure(figure: plt.Figure, figure_dir: Path, name: str) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    figure.savefig(figure_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()

    data_dir = args.output / "data"
    figure_dir = args.output / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(args.cells)
    validate_cells(cells)
    cells["pretraining_updates"] = cells.model_id.map(pretraining_step)
    cells.to_csv(data_dir / "graphsage_matched_saturation_cells.csv", index=False)
    test = cells[cells.split == "test"].copy()

    curve = (
        test.groupby(
            ["pretraining_updates", "label_budget_per_class", "head_updates"], as_index=False
        )
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            cells=("roc_auc", "size"),
        )
        .sort_values(["label_budget_per_class", "head_updates", "pretraining_updates"])
    )
    if set(curve.cells) != {len(TARGETS) * len(LABEL_SEEDS)}:
        raise ValueError("GraphSAGE aggregate curve does not contain every target and label seed")
    curve.to_csv(data_dir / "graphsage_matched_saturation_learning_curves.csv", index=False)

    endpoint = curve[
        (curve.label_budget_per_class == 100) & (curve.head_updates == 100)
    ].sort_values("pretraining_updates")
    endpoint.to_csv(data_dir / "graphsage_matched_saturation_endpoint.csv", index=False)
    positions = np.arange(len(STEPS))
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), sharex=True)
    for axis, metric, label, color in zip(
        axes,
        ("roc_auc_mean", "accuracy_mean", "macro_f1_mean"),
        ("ROC-AUC", "accuracy", "macro-F1"),
        ("#4477AA", "#228833", "#CC6677"),
    ):
        axis.plot(positions, endpoint[metric], marker="o", linewidth=2, color=color)
        axis.set_ylabel(f"test {label}")
        axis.set_xticks(positions, [f"{step:,}" for step in STEPS], rotation=35)
        axis.set_xlabel("native link-prediction updates")
        axis.grid(alpha=0.25)
    figure.suptitle("GraphSAGE pilot-v1 saturation: 100 labels/class, 100 head updates")
    figure.tight_layout()
    save_figure(figure, figure_dir, "graphsage_matched_saturation_endpoint")

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=True)
    colors = dict(zip(UPDATES, plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(UPDATES)))))
    for axis, budget in zip(axes.flat, BUDGETS):
        panel = curve[curve.label_budget_per_class == budget]
        valid_updates = (0,) if budget == 0 else UPDATES
        for update in valid_updates:
            rows = panel[panel.head_updates == update].sort_values("pretraining_updates")
            axis.plot(
                positions, rows.roc_auc_mean, marker="o", color=colors[update],
                label=f"{update} head updates",
            )
        axis.set_title(f"{budget} label{'s' if budget != 1 else ''}/class")
        axis.set_xticks(positions, [f"{step:,}" for step in STEPS], rotation=35)
        axis.grid(alpha=0.25)
    axes[0, 0].set_ylabel("mean test ROC-AUC")
    axes[1, 0].set_ylabel("mean test ROC-AUC")
    axes[1, 0].set_xlabel("native pretraining updates")
    axes[1, 1].set_xlabel("native pretraining updates")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    figure.suptitle("GraphSAGE fixed-head learning curves across the SSL trajectory", y=1.01)
    figure.tight_layout()
    save_figure(figure, figure_dir, "graphsage_matched_saturation_full_grid")

    target_endpoint = (
        test[
            (test.label_budget_per_class == 100) & (test.head_updates == 100)
        ]
        .groupby(["target", "pretraining_updates"], as_index=False)
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
        )
    )
    target_endpoint.to_csv(
        data_dir / "graphsage_matched_saturation_endpoint_by_target.csv", index=False
    )
    figure, axis = plt.subplots(figsize=(7.6, 4.5))
    for target in TARGETS:
        rows = target_endpoint[target_endpoint.target == target].sort_values("pretraining_updates")
        axis.plot(positions, rows.roc_auc_mean, marker="o", label=target)
    axis.set_xticks(positions, [f"{step:,}" for step in STEPS])
    axis.set_xlabel("native link-prediction updates")
    axis.set_ylabel("test ROC-AUC")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    save_figure(figure, figure_dir, "graphsage_matched_saturation_by_target")

    manifest = {
        "source": str(args.cells),
        "rows": int(len(cells)),
        "test_rows": int(len(test)),
        "pretraining_steps": list(STEPS),
        "targets": list(TARGETS),
        "label_seeds": list(LABEL_SEEDS),
        "label_budgets": list(BUDGETS),
        "head_updates": list(UPDATES),
        "training_seeds": 1,
        "protocol": "frozen encoder; shared linear head and sampled nodes",
    }
    (data_dir / "graphsage_matched_saturation_summary.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    delta = float(endpoint.iloc[-1].roc_auc_mean - endpoint.iloc[0].roc_auc_mean)
    print(
        f"GRAPHSAGE_MATCHED_SATURATION_OK rows={len(cells)} "
        f"terminal_minus_init_auc={delta:+.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
