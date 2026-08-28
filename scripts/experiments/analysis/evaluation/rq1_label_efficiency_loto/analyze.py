#!/usr/bin/env python3
"""Validate, aggregate, and plot the completed RQ1 leave-one-family-out experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGETS = ("covid_political", "election2020", "ukr_rus_suspended", "twibot20")
BUDGETS = (1, 10, 100, 1000)
ARMS = ("scratch", "pretrained")
SEEDS = (0, 1, 2)
TARGET_LABELS = {
    "covid_political": "COVID political",
    "election2020": "Election 2020",
    "ukr_rus_suspended": "Ukraine suspended",
    "twibot20": "TwiBot-20",
}
COLORS = {"scratch": "#7f7f7f", "pretrained": "#1f77b4"}


def load_results(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("result.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "target": value["target"],
                "budget_per_class": int(value["budget_per_class"]),
                "arm": value["arm"],
                "seed": int(value["seed"]),
                "selected_best_update": int(value["selected_best_update"]),
                "selected_val_roc_auc": float(value["selected_val_roc_auc"]),
                "updates_run": int(value["updates_run"]),
                "stop_reason": value["stop_reason"],
                "test_roc_auc": float(value["test"]["roc_auc"]),
                "test_accuracy": float(value["test"]["accuracy"]),
                "test_macro_f1": float(value["test"]["macro_f1"]),
                "selected_nodes_fingerprint": value["selected_nodes_fingerprint"],
                "split_fingerprint": value["split_fingerprint"],
                "pretrained_checkpoint_sha256": value["pretrained_checkpoint_sha256"],
                "path": str(path),
            }
        )
    if not rows:
        raise FileNotFoundError(f"no result.json files below {root}")
    return pd.DataFrame(rows)


def validate(frame: pd.DataFrame) -> None:
    expected = {
        (target, budget, arm, seed)
        for target in TARGETS
        for budget in BUDGETS
        for arm in ARMS
        for seed in SEEDS
    }
    keys = set(
        frame[["target", "budget_per_class", "arm", "seed"]]
        .itertuples(index=False, name=None)
    )
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(f"incomplete RQ1 grid: missing={missing[:10]} extra={extra[:10]}")
    if frame.duplicated(["target", "budget_per_class", "arm", "seed"]).any():
        raise ValueError("duplicate RQ1 result cells")
    paired = frame.groupby(["target", "budget_per_class", "seed"])
    for key, rows in paired:
        if rows.selected_nodes_fingerprint.nunique() != 1:
            raise ValueError(f"paired label samples differ for {key}")
        if rows.split_fingerprint.nunique() != 1:
            raise ValueError(f"paired splits differ for {key}")
    pretrained = frame[frame.arm == "pretrained"]
    if (pretrained.pretrained_checkpoint_sha256.str.len() != 64).any():
        raise ValueError("pretrained cells lack checkpoint hashes")
    if (frame[frame.arm == "scratch"].pretrained_checkpoint_sha256 != "").any():
        raise ValueError("scratch cells unexpectedly reference pretrained checkpoints")


def save_figure(fig, root: Path, name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    data_dir = args.output / "data"
    figure_dir = args.output / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame = load_results(args.results_root)
    validate(frame)
    frame.to_csv(data_dir / "rq1_cells.csv", index=False)

    summary = (
        frame.groupby(["target", "budget_per_class", "arm"], as_index=False)
        .agg(
            test_roc_auc_mean=("test_roc_auc", "mean"),
            test_roc_auc_std=("test_roc_auc", "std"),
            selected_update_median=("selected_best_update", "median"),
            updates_run_median=("updates_run", "median"),
        )
    )
    summary.to_csv(data_dir / "rq1_summary.csv", index=False)

    paired = frame.pivot(
        index=["target", "budget_per_class", "seed"],
        columns="arm",
        values="test_roc_auc",
    ).reset_index()
    paired["pretrained_minus_scratch"] = paired.pretrained - paired.scratch
    paired.to_csv(data_dir / "rq1_paired_deltas.csv", index=False)
    delta_summary = (
        paired.groupby(["target", "budget_per_class"], as_index=False)
        .agg(
            delta_mean=("pretrained_minus_scratch", "mean"),
            delta_std=("pretrained_minus_scratch", "std"),
            positive_seed_fraction=("pretrained_minus_scratch", lambda x: float((x > 0).mean())),
        )
    )
    delta_summary.to_csv(data_dir / "rq1_delta_summary.csv", index=False)

    log_budget = np.log10(np.asarray(BUDGETS, dtype=float))
    aulc_rows = []
    for (target, arm, seed), rows in frame.groupby(["target", "arm", "seed"]):
        values = rows.set_index("budget_per_class").loc[list(BUDGETS), "test_roc_auc"].to_numpy()
        aulc_rows.append(
            {
                "target": target,
                "arm": arm,
                "seed": seed,
                "label_efficiency_aulc": float(np.trapz(values, log_budget) / 3.0),
            }
        )
    aulc = pd.DataFrame(aulc_rows)
    aulc.to_csv(data_dir / "rq1_label_efficiency_aulc.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4), sharex=True, sharey=True)
    x = np.arange(len(BUDGETS))
    for panel, (axis, target) in enumerate(zip(axes.flat, TARGETS)):
        for arm in ARMS:
            rows = summary[(summary.target == target) & (summary.arm == arm)].set_index(
                "budget_per_class"
            )
            mean = rows.loc[list(BUDGETS), "test_roc_auc_mean"].to_numpy()
            std = rows.loc[list(BUDGETS), "test_roc_auc_std"].to_numpy()
            axis.errorbar(
                x,
                mean,
                yerr=std,
                color=COLORS[arm],
                marker="o" if arm == "pretrained" else "s",
                linewidth=2,
                capsize=3,
                label="Multi-graph SSL" if arm == "pretrained" else "Scratch",
            )
        axis.set_title(f"{chr(ord('a') + panel)}  {TARGET_LABELS[target]}", loc="left")
        axis.set_xticks(x, [str(value) for value in BUDGETS])
        axis.grid(alpha=0.22)
    for axis in axes[-1]:
        axis.set_xlabel("Labeled examples per class")
    for axis in axes[:, 0]:
        axis.set_ylabel("Test ROC-AUC")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("RQ1: target-family-excluded pretraining versus scratch", y=1.02)
    fig.tight_layout()
    save_figure(fig, figure_dir, "rq1_label_efficiency_by_target")

    overall = (
        paired.groupby("budget_per_class", as_index=False)
        .agg(
            delta_mean=("pretrained_minus_scratch", "mean"),
            delta_std=("pretrained_minus_scratch", "std"),
            positive_cell_fraction=("pretrained_minus_scratch", lambda x: float((x > 0).mean())),
        )
    )
    overall.to_csv(data_dir / "rq1_overall.csv", index=False)
    print(overall.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
