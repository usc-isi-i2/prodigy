#!/usr/bin/env python3
"""Validate and plot SAMGPT all-nine GraphCL-to-CLS checkpoint trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
UPDATES = (20, 60, 180, 500)
SEEDS = (39, 40, 41)
TARGETS = (
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
    "facebook_page_category_top30",
    "facebook_admin_country_top30",
    "facebook_verified",
    "cora",
    "pubmed",
)
LABELS = {
    "covid_political": "COVID Political",
    "election2020": "Election 2020",
    "ukr_rus_suspended": "Ukraine Suspended",
    "twibot20": "TwiBot-20",
    "facebook_page_category_top30": "Facebook category",
    "facebook_admin_country_top30": "Facebook country",
    "facebook_verified": "Facebook verified",
    "cora": "Cora",
    "pubmed": "PubMed",
}


def load_cells(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("model_id") != "all9" or not value.get("complete"):
            continue
        seed = int(value["seed"])
        update = int(value["checkpoint_update"])
        for target, metrics in value["targets"].items():
            rows.append(
                {
                    "seed": seed,
                    "checkpoint_update": update,
                    "target": target,
                    "roc_auc": float(metrics["roc_auc_mean"]),
                    "accuracy": float(metrics["accuracy_mean"]),
                    "roc_auc_episode_std": float(metrics["roc_auc_std"]),
                    "accuracy_episode_std": float(metrics["accuracy_std"]),
                    "episode_fingerprint": metrics["episode_fingerprint"],
                    "checkpoint_sha256": value["checkpoint_sha256"],
                    "source_file": str(path),
                }
            )
    frame = pd.DataFrame(rows)
    expected = {
        (seed, update, target)
        for seed in SEEDS
        for update in UPDATES
        for target in TARGETS
    }
    observed = set(zip(frame.seed, frame.checkpoint_update, frame.target, strict=True))
    if observed != expected or len(frame) != len(expected):
        raise ValueError(
            f"SAMGPT saturation coverage mismatch: rows={len(frame)} "
            f"missing={sorted(expected - observed)} extra={sorted(observed - expected)}"
        )
    fingerprints = frame.groupby("target").episode_fingerprint.nunique()
    if not fingerprints.eq(1).all():
        raise ValueError(f"episode streams differ across checkpoints/seeds: {fingerprints.to_dict()}")
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["target", "checkpoint_update"], as_index=False)
        .agg(
            training_seeds=("seed", "nunique"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_sample_std=("roc_auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sample_std=("accuracy", "std"),
        )
        .sort_values(["target", "checkpoint_update"])
    )


def plot(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(11.6, 8.5), sharex=True, sharey=True, constrained_layout=True)
    for axis, target in zip(axes.flat, TARGETS, strict=True):
        rows = summary[summary.target.eq(target)].sort_values("checkpoint_update")
        x = rows.checkpoint_update.to_numpy(dtype=float)
        mean = rows.roc_auc_mean.to_numpy(dtype=float)
        std = rows.roc_auc_sample_std.fillna(0).to_numpy(dtype=float)
        axis.plot(x, mean, color="#2a9d8f", marker="o", linewidth=2)
        axis.fill_between(x, mean - std, mean + std, color="#2a9d8f", alpha=0.18)
        axis.set_title(LABELS[target], fontsize=10)
        axis.set_xscale("log")
        axis.set_xticks(UPDATES, [str(value) for value in UPDATES])
        axis.set_ylim(0.45, 1.0)
        axis.grid(axis="y", alpha=0.22)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("GraphCL updates")
    for axis in axes[:, 0]:
        axis.set_ylabel("Downstream ROC-AUC")
    fig.suptitle("SAMGPT all-nine GraphCL→CLS saturation (three training seeds)", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "samgpt_all9_saturation_raw",
    )
    args = parser.parse_args()
    cells = load_cells(args.input)
    cells.to_csv(ROOT / "data" / "samgpt_all9_saturation_cells.csv", index=False)
    summary = summarize(cells)
    summary.to_csv(ROOT / "data" / "samgpt_all9_saturation_summary.csv", index=False)
    plot(summary, ROOT / "figures" / "samgpt_all9_saturation")
    print(f"SAMGPT_SATURATION_OK rows={len(cells)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
