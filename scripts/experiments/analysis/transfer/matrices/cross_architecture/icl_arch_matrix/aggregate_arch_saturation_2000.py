#!/usr/bin/env python3
"""Validate, aggregate, and plot the matched 2,000-step architecture sweep."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARCHITECTURES = ("prodigy", "vision", "gilt")
STEPS = (0, 20, 60, 100, 300, 900, 2000)
TRAINING_SEEDS = (0, 1, 2)
EVAL_OFFSETS = (0, 1, 2)
MODELS = (
    "ss_covid_political",
    "ss_election2020",
    "ss_ukr_rus_suspended",
    "ss_twibot20",
    "ss_facebook_page_reference",
)
TARGETS = (
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
    "facebook_page_reference",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--steps", default="0,20,60,100,300,900,2000")
    parser.add_argument("--training-seeds", default="0,1,2")
    parser.add_argument("--eval-offsets", default="0,1,2")
    return parser.parse_args()


def load_rows(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    row["result_file"] = str(path)
                    rows.append(row)
    return rows


def validate(rows: list[dict]) -> None:
    nm = [row for row in rows if row["task"] == "neighbor_matching"]
    cls = [row for row in rows if row["task"] == "classification"]
    expected_nm = len(ARCHITECTURES) * len(MODELS) * len(TRAINING_SEEDS) * len(EVAL_OFFSETS) * len(STEPS)
    expected_cls = len(ARCHITECTURES) * len(TRAINING_SEEDS) * len(EVAL_OFFSETS) * len(TARGETS) + (
        len(ARCHITECTURES) * len(MODELS) * len(TRAINING_SEEDS) * len(EVAL_OFFSETS)
        * (len(STEPS) - 1) * len(TARGETS)
    )
    if len(nm) != expected_nm or len(cls) != expected_cls:
        raise ValueError(
            f"incomplete results: NM {len(nm)}/{expected_nm}, CLS {len(cls)}/{expected_cls}"
        )
    nm_keys = {
        (row["architecture"], row["model_id"], int(row["training_seed"]),
         int(row["eval_episode_seed_offset"]), int(row["checkpoint_step"]))
        for row in nm
    }
    cls_keys = {
        (row["architecture"], row["model_id"], int(row.get("training_seed", row["seed"])),
         int(row["eval_episode_seed_offset"]), int(row["checkpoint_step"]), row["dataset"])
        for row in cls
    }
    if len(nm_keys) != expected_nm or len(cls_keys) != expected_cls:
        raise ValueError("duplicate or missing evaluation keys")
    fingerprints = defaultdict(set)
    for row in rows:
        fingerprint = row.get("episode_fingerprint", "")
        if fingerprint:
            fingerprints[
                (row["task"], row["dataset"], int(row["eval_episode_seed_offset"]))
            ].add(fingerprint)
    # PRODIGY's native-NM evaluator does not currently export episode fingerprints.
    # Validate every fingerprint that is available without treating that known metadata
    # omission as evidence that the episode streams drifted.
    drift = {key: values for key, values in fingerprints.items() if len(values) != 1}
    if drift:
        raise ValueError(f"episode fingerprint drift: {drift}")


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["architecture"], row["task"], int(row["checkpoint_step"]))].append(
            float(row["roc_auc"])
        )
    summary = []
    for architecture in ARCHITECTURES:
        for task in ("neighbor_matching", "classification"):
            for step in STEPS:
                values = np.asarray(groups[(architecture, task, step)], dtype=float)
                if not values.size:
                    raise ValueError(f"empty aggregate: {architecture}/{task}/{step}")
                summary.append({
                    "architecture": architecture,
                    "task": task,
                    "checkpoint_step": step,
                    "mean_roc_auc": float(values.mean()),
                    "std_roc_auc": float(values.std(ddof=1)),
                    "n": int(values.size),
                })
    return summary


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    fields.extend(sorted(set().union(*(row.keys() for row in rows)) - set(fields)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def plot(summary: list[dict], output_root: Path) -> None:
    by_key = {(row["architecture"], row["task"], row["checkpoint_step"]): row for row in summary}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)
    positions = np.arange(len(STEPS))
    for axis, architecture in zip(axes, ARCHITECTURES):
        for task, label, color in (
            ("neighbor_matching", "Mean NM AUC", "#4477AA"),
            ("classification", "Mean classification AUC", "#EE6677"),
        ):
            means = [by_key[(architecture, task, step)]["mean_roc_auc"] for step in STEPS]
            axis.plot(positions, means, linewidth=2.3, color=color, label=label)
        axis.set_title(architecture.upper())
        axis.set_xticks(positions, [str(step) if step < 1000 else "2k" for step in STEPS])
        axis.set_xlabel("Training checkpoint")
        axis.set_ylabel("Mean ROC-AUC")
        axis.grid(axis="y", alpha=0.25)
        values = [line.get_ydata() for line in axis.lines]
        lo, hi = min(map(np.min, values)), max(map(np.max, values))
        pad = max(0.02, (hi - lo) * 0.1)
        axis.set_ylim(max(0, lo - pad), min(1, hi + pad))
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(
        f"Matched {max(STEPS):,}-step trajectories: five sources × "
        f"{len(TRAINING_SEEDS)} training seed(s) × {len(EVAL_OFFSETS)} eval sample(s)"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_root / "mean_nm_cls_trajectory.png", dpi=180)
    fig.savefig(output_root / "mean_nm_cls_trajectory.pdf")
    plt.close(fig)


def main() -> int:
    global STEPS, TRAINING_SEEDS, EVAL_OFFSETS
    args = parse_args()
    STEPS = tuple(int(value) for value in args.steps.split(","))
    TRAINING_SEEDS = tuple(int(value) for value in args.training_seeds.split(","))
    EVAL_OFFSETS = tuple(int(value) for value in args.eval_offsets.split(","))
    rows = load_rows(args.results_root)
    validate(rows)
    summary = aggregate(rows)
    write_tsv(args.output_root / "all_results.tsv", rows)
    write_tsv(args.output_root / "trajectory_summary.tsv", summary)
    plot(summary, args.output_root)
    (args.output_root / "COMPLETE.json").write_text(
        json.dumps({"rows": len(rows), "summary_rows": len(summary)}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "summary_rows": len(summary)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
