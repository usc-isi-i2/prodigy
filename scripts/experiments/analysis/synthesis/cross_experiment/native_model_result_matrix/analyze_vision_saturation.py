#!/usr/bin/env python3
"""Validate and plot the three-seed VISION all-nine SSL-to-CLS trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
STEPS = (100, 300, 900, 2500)
SEEDS = (0, 1, 2)
TARGETS = (
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
    "facebook_page_reference",
)
LABELS = {
    "covid_political": "COVID Political",
    "election2020": "Election 2020",
    "ukr_rus_suspended": "Ukraine Suspended",
    "twibot20": "TwiBot-20",
    "facebook_page_reference": "Facebook Page",
}


def load_rows(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    frame = pd.DataFrame(rows)
    expected = {
        (seed, step, target) for seed in SEEDS for step in STEPS for target in TARGETS
    }
    observed = set(
        zip(frame["training_seed"], frame["checkpoint_step"], frame["dataset"], strict=True)
    )
    if observed != expected or len(frame) != len(expected):
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"trajectory coverage mismatch: rows={len(frame)} missing={missing} extra={extra}")
    if set(frame["architecture"]) != {"vision"} or set(frame["task"]) != {"classification"}:
        raise ValueError("input contains a non-VISION or non-classification result")
    fingerprints = frame.groupby("dataset")["episode_fingerprint"].nunique()
    if not fingerprints.eq(1).all():
        raise ValueError(f"episode streams differ across checkpoints/seeds: {fingerprints.to_dict()}")
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    summary = (
        frame.groupby(["dataset", "checkpoint_step"], as_index=False)
        .agg(
            training_seeds=("training_seed", "nunique"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_sample_std=("roc_auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sample_std=("accuracy", "std"),
            macro_f1_mean=("f1", "mean"),
            macro_f1_sample_std=("f1", "std"),
        )
        .sort_values(["dataset", "checkpoint_step"])
    )
    macro = (
        frame.groupby(["training_seed", "checkpoint_step"], as_index=False)[
            ["roc_auc", "accuracy", "f1"]
        ]
        .mean()
        .groupby("checkpoint_step", as_index=False)
        .agg(
            training_seeds=("training_seed", "nunique"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_sample_std=("roc_auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sample_std=("accuracy", "std"),
            macro_f1_mean=("f1", "mean"),
            macro_f1_sample_std=("f1", "std"),
        )
    )
    macro.insert(0, "dataset", "macro_target_mean")
    return pd.concat([summary, macro], ignore_index=True)


def plot(summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.8), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for index, target in enumerate((*TARGETS, "macro_target_mean")):
        rows = summary[summary["dataset"].eq(target)].sort_values("checkpoint_step")
        x = rows["checkpoint_step"].to_numpy(dtype=float)
        mean = rows["roc_auc_mean"].to_numpy(dtype=float)
        std = rows["roc_auc_sample_std"].fillna(0).to_numpy(dtype=float)
        label = "Five-target mean" if target == "macro_target_mean" else LABELS[target]
        width = 3.2 if target == "macro_target_mean" else 1.8
        alpha = 1.0 if target == "macro_target_mean" else 0.82
        color = "black" if target == "macro_target_mean" else colors(index)
        ax.plot(x, mean, marker="o", linewidth=width, alpha=alpha, color=color, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.10)
    ax.set_xscale("log")
    ax.set_xticks(STEPS, [str(step) for step in STEPS])
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("Native feature-similarity optimizer updates (fixed compute)")
    ax.set_ylabel("Downstream CLS ROC-AUC")
    ax.set_title("VISION all-nine SSL→CLS saturation (three training seeds)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=8.5, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "vision_all9_saturation.jsonl")
    args = parser.parse_args()
    frame = load_rows(args.input)
    summary = summarize(frame)
    summary.to_csv(ROOT / "data" / "vision_all9_saturation_summary.csv", index=False)
    plot(summary, ROOT / "figures" / "vision_all9_saturation")
    print(f"VISION_SATURATION_OK rows={len(frame)} logical_cells={len(frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
