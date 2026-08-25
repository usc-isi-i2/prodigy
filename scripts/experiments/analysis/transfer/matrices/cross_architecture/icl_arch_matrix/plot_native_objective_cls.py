#!/usr/bin/env python3
"""Consolidate and plot native-objective classification trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARCHITECTURES = ("prodigy", "vision", "gilt")
STEPS = (0, 20, 60, 100, 300, 900)
TARGETS = (
    ("covid_political", "COVID political"),
    ("election2020", "Election 2020"),
    ("ukr_rus_suspended", "UKR/RUS suspended"),
    ("twibot20", "TwiBot-20"),
    ("facebook_page_reference", "Facebook pages"),
)
PROTOCOLS = {
    "prodigy": "native_neighbor_matching",
    "vision": "native_feature_similarity",
    "gilt": "native_source_classification",
}
COLORS = {"prodigy": "#4477AA", "vision": "#228833", "gilt": "#CC6677"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prodigy-results", required=True, type=Path)
    parser.add_argument("--native-results-root", required=True, type=Path)
    parser.add_argument("--output-data", required=True, type=Path)
    parser.add_argument("--output-figures", required=True, type=Path)
    return parser.parse_args()


def load_prodigy(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    keep = []
    for row in rows:
        if row["architecture"] != "prodigy" or row["task"] != "classification":
            continue
        item = dict(row)
        item["checkpoint_step"] = int(row["checkpoint_step"])
        item["roc_auc"] = float(row["roc_auc"])
        item["protocol"] = PROTOCOLS["prodigy"]
        keep.append(item)
    return keep


def load_native(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["architecture"] not in {"vision", "gilt"}:
                raise ValueError(f"unexpected architecture in {path}: {row['architecture']}")
            row["result_file"] = str(path)
            row["protocol"] = PROTOCOLS[row["architecture"]]
            rows.append(row)
    return rows


def validate(rows: list[dict]) -> None:
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["architecture"], int(row["checkpoint_step"]), row["dataset"])].append(row)

    for architecture in ARCHITECTURES:
        for step in STEPS:
            for target, _ in TARGETS:
                cell = grouped[(architecture, step, target)]
                expected = 1 if step == 0 else 5
                if len(cell) != expected:
                    raise ValueError(
                        f"{architecture}/{step}/{target}: got {len(cell)} rows, expected {expected}"
                    )
                fingerprints = {row["episode_fingerprint"] for row in cell}
                if len(fingerprints) != 1:
                    raise ValueError(f"episode fingerprint mismatch for {architecture}/{step}/{target}")


def write_data(rows: list[dict], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fields = [
        "architecture", "protocol", "model_id", "sources", "dataset",
        "checkpoint_step", "roc_auc", "accuracy", "f1", "seed",
        "training_seed", "eval_episode_seed_offset", "episodes", "queries",
        "n_way", "n_shot", "n_query", "episode_fingerprint", "baseline",
        "result_file",
    ]
    ordered = sorted(
        rows,
        key=lambda row: (
            ARCHITECTURES.index(row["architecture"]),
            int(row["checkpoint_step"]),
            row["dataset"],
            row["model_id"],
        ),
    )
    with (output_root / "classification_all.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)

    summary = []
    for architecture in ARCHITECTURES:
        for step in STEPS:
            for target, _ in TARGETS:
                values = [
                    float(row["roc_auc"])
                    for row in rows
                    if row["architecture"] == architecture
                    and int(row["checkpoint_step"]) == step
                    and row["dataset"] == target
                ]
                summary.append(
                    {
                        "architecture": architecture,
                        "protocol": PROTOCOLS[architecture],
                        "checkpoint_step": step,
                        "target": target,
                        "n_models": len(values),
                        "mean_roc_auc": float(np.mean(values)),
                        "std_roc_auc_across_source_models": float(np.std(values, ddof=1)) if len(values) > 1 else "",
                    }
                )
    fields = list(summary[0])
    with (output_root / "classification_summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(summary)


def trajectory(rows: list[dict], architecture: str, target: str) -> list[float]:
    return [
        float(np.mean([
            float(row["roc_auc"])
            for row in rows
            if row["architecture"] == architecture
            and int(row["checkpoint_step"]) == step
            and row["dataset"] == target
        ]))
        for step in STEPS
    ]


def padded_limits(values: list[float]) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    pad = max(0.015, (hi - lo) * 0.09)
    return max(0.0, lo - pad), min(1.0, hi + pad)


def plot_by_target(rows: list[dict], architecture: str, output_root: Path) -> None:
    positions = np.arange(len(STEPS))
    curves = {target: trajectory(rows, architecture, target) for target, _ in TARGETS}
    y_limits = padded_limits([value for curve in curves.values() for value in curve])
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(18, 3.55), sharey=True)
    for index, (axis, (target, title)) in enumerate(zip(axes, TARGETS)):
        axis.plot(
            positions,
            curves[target],
            color=COLORS[architecture],
            marker="o",
            markersize=4.5,
            linewidth=2.2,
        )
        axis.set_title(title)
        axis.set_xticks(positions, [str(step) for step in STEPS])
        axis.set_xlabel("Training checkpoint")
        axis.set_ylim(*y_limits)
        axis.grid(axis="y", alpha=0.25)
        if index == 0:
            axis.set_ylabel("Classification ROC-AUC")
    title = {
        "prodigy": "PRODIGY native NM pretraining",
        "vision": "VISION native feature-similarity pretraining",
        "gilt": "GILT source-confined native classification pretraining",
    }[architecture]
    fig.suptitle(f"{title}: target classification (mean over source models)", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    stem = output_root / f"{architecture}_native_objective_cls_by_target_900_seed0"
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_overall(rows: list[dict], output_root: Path) -> None:
    positions = np.arange(len(STEPS))
    curves = {}
    for architecture in ARCHITECTURES:
        curves[architecture] = [
            float(np.mean([
                float(row["roc_auc"])
                for row in rows
                if row["architecture"] == architecture
                and int(row["checkpoint_step"]) == step
            ]))
            for step in STEPS
        ]
    fig, axis = plt.subplots(figsize=(7.4, 4.5))
    for architecture in ARCHITECTURES:
        axis.plot(
            positions,
            curves[architecture],
            color=COLORS[architecture],
            marker="o",
            linewidth=2.3,
            label=architecture.upper(),
        )
    axis.set_xticks(positions, [str(step) for step in STEPS])
    axis.set_xlabel("Training checkpoint")
    axis.set_ylabel("Mean classification ROC-AUC")
    axis.set_ylim(*padded_limits([value for curve in curves.values() for value in curve]))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    axis.set_title("Native-objective training: classification transfer")
    fig.tight_layout()
    stem = output_root / "native_objective_cls_mean_900_seed0"
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = load_prodigy(args.prodigy_results) + load_native(args.native_results_root)
    validate(rows)
    args.output_figures.mkdir(parents=True, exist_ok=True)
    write_data(rows, args.output_data)
    for architecture in ARCHITECTURES:
        plot_by_target(rows, architecture, args.output_figures)
    plot_overall(rows, args.output_figures)
    print(f"rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
