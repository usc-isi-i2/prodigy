#!/usr/bin/env python3
"""Validate and plot the three-seed VISION native mixture-diversity ladder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[5]))

from scripts.experiments.setup.vision_native_mixture_finalcore.mixture_plan import (
    RUNGS,
    build_mixture_models,
)


CHECKPOINTS = (100, 300, 900, 2500)
SEEDS = (0, 1, 2)
TARGETS = (
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
    "facebook_page_reference",
)
ORDERS = ("A", "B", "C")
COLORS = {"A": "#4477AA", "B": "#CC6677", "C": "#228833"}
TARGET_LABELS = {
    "covid_political": "COVID Political",
    "election2020": "Election 2020",
    "ukr_rus_suspended": "Ukraine Suspended",
    "twibot20": "TwiBot-20",
    "facebook_page_reference": "Facebook Page",
}
EXPECTED_PHYSICAL_CELLS = 13 * len(CHECKPOINTS) * len(TARGETS) * len(SEEDS)
EXPECTED_EXPANDED_CELLS = len(ORDERS) * len(RUNGS) * len(CHECKPOINTS) * len(TARGETS) * len(SEEDS)


def read_jsonl_roots(roots: list[Path]) -> pd.DataFrame:
    rows: list[dict] = []
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        paths.extend(sorted(root.rglob("*.jsonl")) if root.is_dir() else [root])
    if not paths:
        raise FileNotFoundError(f"no JSONL files under {roots}")
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return pd.DataFrame(rows)


def load_cells(mixture_roots: list[Path], all9_root: Path) -> pd.DataFrame:
    mixture = read_jsonl_roots(mixture_roots)
    all9 = read_jsonl_roots([all9_root])
    all9 = all9[all9["model_id"].eq("all9") & all9["training_seed"].isin(SEEDS)]

    models = build_mixture_models()
    expected_mixture = {model.model_id for model in models if model.model_id != "all9"}
    if set(mixture["model_id"]) != expected_mixture:
        raise ValueError(
            "VISION mixture model registry mismatch: "
            f"observed={sorted(set(mixture['model_id']))} expected={sorted(expected_mixture)}"
        )
    if set(mixture["training_seed"]) != set(SEEDS):
        raise ValueError(
            f"expected training seeds {SEEDS} in non-all-nine mixtures, "
            f"got {sorted(set(mixture['training_seed']))}"
        )
    if set(all9["training_seed"]) != set(SEEDS):
        raise ValueError(
            f"expected training seeds {SEEDS} in all-nine controls, "
            f"got {sorted(set(all9['training_seed']))}"
        )
    frame = pd.concat((mixture, all9), ignore_index=True)

    required = {
        "architecture", "task", "model_id", "sources", "training_seed",
        "checkpoint_step", "dataset", "episode_fingerprint", "roc_auc",
        "accuracy", "f1",
    }
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"missing required result fields: {sorted(missing_columns)}")
    if set(frame["architecture"]) != {"vision"} or set(frame["task"]) != {"classification"}:
        raise ValueError("mixture input contains non-VISION or non-CLS rows")
    if set(frame["training_seed"]) != set(SEEDS):
        raise ValueError(f"expected training seeds {SEEDS}, got {sorted(set(frame['training_seed']))}")
    if set(frame["checkpoint_step"]) != set(CHECKPOINTS):
        raise ValueError("VISION mixture checkpoints are incomplete")
    if set(frame["dataset"]) != set(TARGETS):
        raise ValueError("VISION mixture target panel changed")

    expected_sources = {model.model_id: frozenset(model.sources) for model in models}
    bad_sources = frame[
        frame.apply(
            lambda row: frozenset(row["sources"]) != expected_sources.get(row["model_id"], frozenset()),
            axis=1,
        )
    ]
    if not bad_sources.empty:
        raise ValueError(f"source registry drift for models {sorted(bad_sources.model_id.unique())}")

    keys = ["model_id", "checkpoint_step", "dataset", "training_seed"]
    if frame.duplicated(keys).any() or len(frame) != EXPECTED_PHYSICAL_CELLS:
        raise ValueError(
            f"expected {EXPECTED_PHYSICAL_CELLS} unique physical cells, got {len(frame)}"
        )
    expected_keys = {
        (model.model_id, step, target, seed)
        for model in models
        for step in CHECKPOINTS
        for target in TARGETS
        for seed in SEEDS
    }
    observed_keys = set(frame[keys].itertuples(index=False, name=None))
    if observed_keys != expected_keys:
        raise ValueError(
            "VISION mixture coverage mismatch: "
            f"missing={len(expected_keys - observed_keys)} extra={len(observed_keys - expected_keys)}"
        )
    fingerprints = frame.groupby("dataset")["episode_fingerprint"].nunique()
    if not fingerprints.eq(1).all():
        raise ValueError(f"downstream episode drift: {fingerprints.to_dict()}")
    return frame.sort_values(keys).reset_index(drop=True)


def expand_orders(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {model.model_id: model.aliases for model in build_mixture_models()}
    rows = []
    for row in frame.to_dict("records"):
        for alias in aliases[row["model_id"]]:
            _, order, rung_text = alias.split(":")
            rows.append({**row, "order": order, "rung": int(rung_text)})
    expanded = pd.DataFrame(rows)
    keys = ["order", "rung", "checkpoint_step", "dataset", "training_seed"]
    if expanded.duplicated(keys).any() or len(expanded) != EXPECTED_EXPANDED_CELLS:
        raise ValueError(
            f"expected {EXPECTED_EXPANDED_CELLS} unique order-expanded cells, got {len(expanded)}"
        )
    return expanded.sort_values(keys).reset_index(drop=True)


def summarize(expanded: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_target = (
        expanded.groupby(
            ["checkpoint_step", "order", "rung", "dataset", "training_seed"],
            as_index=False,
        )[["roc_auc", "accuracy", "f1"]]
        .mean()
    )
    per_seed = (
        per_target.groupby(
            ["checkpoint_step", "order", "rung", "training_seed"], as_index=False
        )[["roc_auc", "accuracy", "f1"]]
        .mean()
    )
    summary = (
        per_seed.groupby(["checkpoint_step", "order", "rung"], as_index=False)
        .agg(
            training_seeds=("training_seed", "nunique"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_min=("roc_auc", "min"),
            roc_auc_max=("roc_auc", "max"),
            roc_auc_sample_std=("roc_auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_min=("accuracy", "min"),
            accuracy_max=("accuracy", "max"),
            f1_mean=("f1", "mean"),
            f1_min=("f1", "min"),
            f1_max=("f1", "max"),
        )
    )
    if not summary["training_seeds"].eq(len(SEEDS)).all():
        raise ValueError("a summarized ladder point does not contain all three training seeds")
    return per_target, summary


def plot_trajectory(summary: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for axis, step in zip(axes, CHECKPOINTS, strict=True):
        subset = summary[summary.checkpoint_step.eq(step)]
        for order in ORDERS:
            line = subset[subset.order.eq(order)].sort_values("rung")
            x = line.rung.to_numpy(dtype=float)
            mean = line.roc_auc_mean.to_numpy(dtype=float)
            low = line.roc_auc_min.to_numpy(dtype=float)
            high = line.roc_auc_max.to_numpy(dtype=float)
            axis.fill_between(x, low, high, color=COLORS[order], alpha=0.13, linewidth=0)
            axis.plot(x, mean, marker="o", linewidth=2, color=COLORS[order], label=f"order {order}")
        axis.set_title(f"{step:,} updates")
        axis.set_xticks(RUNGS)
        axis.set_xlabel("source graphs")
        axis.grid(alpha=0.25)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_ylabel("mean downstream ROC-AUC\n(five fixed CLS targets)")
    axes[-1].legend(frameon=False, loc="best")
    figure.suptitle("VISION mixture diversity across three training seeds")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_terminal_targets(per_target: pd.DataFrame, output: Path) -> None:
    terminal = per_target[per_target.checkpoint_step.eq(2500)]
    target_summary = (
        terminal.groupby(["order", "rung", "dataset"], as_index=False)
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_min=("roc_auc", "min"),
            roc_auc_max=("roc_auc", "max"),
        )
    )
    figure, axes = plt.subplots(2, 3, figsize=(11, 7), sharex=True, sharey=True)
    for axis, target in zip(axes.flat, TARGETS):
        subset = target_summary[target_summary.dataset.eq(target)]
        for order in ORDERS:
            line = subset[subset.order.eq(order)].sort_values("rung")
            x = line.rung.to_numpy(dtype=float)
            mean = line.roc_auc_mean.to_numpy(dtype=float)
            low = line.roc_auc_min.to_numpy(dtype=float)
            high = line.roc_auc_max.to_numpy(dtype=float)
            axis.fill_between(x, low, high, color=COLORS[order], alpha=0.13, linewidth=0)
            axis.plot(x, mean, marker="o", color=COLORS[order], label=order)
        axis.set_title(TARGET_LABELS[target])
        axis.set_xticks(RUNGS)
        axis.grid(alpha=0.25)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes.flat[-1].axis("off")
    axes[0, 0].set_ylabel("ROC-AUC")
    axes[1, 0].set_ylabel("ROC-AUC")
    for axis in axes[1, :2]:
        axis.set_xlabel("source graphs")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, [f"order {value}" for value in labels], frameon=False)
    figure.suptitle("VISION terminal mixture ladders (mean and seed range)")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mixture-root", type=Path, action="append", dest="mixture_roots",
        help="Repeat for the seed-0, seed-1, and seed-2 non-all-nine result roots.",
    )
    parser.add_argument(
        "--all9-root", type=Path,
        default=ROOT / "data" / "vision_all9_saturation_raw",
    )
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    mixture_roots = args.mixture_roots or [
        ROOT / "data" / "vision_native_mixture_raw",
        ROOT / "data" / "vision_native_mixture_seed1_raw",
        ROOT / "data" / "vision_native_mixture_seed2_raw",
    ]
    frame = load_cells(mixture_roots, args.all9_root)
    expanded = expand_orders(frame)
    per_target, summary = summarize(expanded)

    data = args.output / "data"
    figures = args.output / "figures"
    data.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data / "vision_native_mixture_three_seed_physical_cells.csv", index=False)
    expanded.to_csv(data / "vision_native_mixture_three_seed_expanded_cells.csv", index=False)
    per_target.to_csv(data / "vision_native_mixture_three_seed_per_target.csv", index=False)
    summary.to_csv(data / "vision_native_mixture_three_seed_summary.csv", index=False)
    plot_trajectory(summary, figures / "vision_mixture_diversity_three_seed_trajectory")
    plot_terminal_targets(per_target, figures / "vision_mixture_diversity_three_seed_terminal_targets")
    print(
        "VISION_MIXTURE_THREE_SEED_OK "
        f"physical_cells={len(frame)} order_expanded_cells={len(expanded)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
