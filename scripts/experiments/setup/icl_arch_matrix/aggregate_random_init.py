#!/usr/bin/env python3
"""Validate the 12-cell random-initialization architecture control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiments.setup.icl_arch_matrix.aggregate_results import (
    ARCHITECTURES,
    TARGETS,
)


def parse_args():
    parser = argparse.ArgumentParser()
    for architecture in ARCHITECTURES:
        parser.add_argument(f"--{architecture}", required=True)
    parser.add_argument("--trained-reference", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    args = parse_args()
    input_paths = {
        architecture: Path(getattr(args, architecture))
        for architecture in ARCHITECTURES
    }
    rows: list[dict] = []
    for architecture, path in input_paths.items():
        loaded = read_jsonl(path)
        if len(loaded) != len(TARGETS):
            raise ValueError(f"expected four {architecture} rows, got {len(loaded)}")
        for row in loaded:
            if row["architecture"] != architecture:
                raise ValueError(f"architecture mismatch in {path}")
            if row["model_id"] != "random_init" or row.get("baseline") != "random_init":
                raise ValueError(f"non-random-init row in {path}: {row}")
            if int(row["checkpoint_step"]) != 0 or int(row["seed"]) != 0:
                raise ValueError(f"invalid random-init step/seed in {path}: {row}")
            if row["sources"] != []:
                raise ValueError(f"random-init baseline must not claim sources: {row}")
            for metric in ("roc_auc", "accuracy", "f1"):
                value = float(row[metric])
                if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"invalid {metric} in {path}: {value}")
        rows.extend(loaded)

    expected = {
        (architecture, target)
        for architecture in ARCHITECTURES
        for target in TARGETS
    }
    observed = [(row["architecture"], row["dataset"]) for row in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError(f"random-init grid mismatch: {observed}")

    trained = pd.read_csv(args.trained_reference)
    if len(trained) != 372:
        raise ValueError(f"expected 372 trained reference rows, got {len(trained)}")
    if set(trained.architecture) != set(ARCHITECTURES) or set(trained.dataset) != set(TARGETS):
        raise ValueError("trained reference architecture/target registry mismatch")
    trained_fingerprints = trained.groupby("dataset").episode_fingerprint.unique()
    for target in TARGETS:
        fingerprints = {row["episode_fingerprint"] for row in rows if row["dataset"] == target}
        if len(fingerprints) != 1:
            raise ValueError(f"architectures consumed different {target} episodes")
        reference = trained_fingerprints.loc[target]
        if len(reference) != 1 or next(iter(fingerprints)) != reference[0]:
            raise ValueError(f"random-init fingerprint differs from trained matrix on {target}")

    rows.sort(
        key=lambda row: (
            ARCHITECTURES.index(row["architecture"]),
            TARGETS.index(row["dataset"]),
        )
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    long_fields = [
        "architecture", "model_id", "sources", "seed", "checkpoint_step", "baseline",
        "task", "dataset", "n_way", "n_shot", "n_query", "episodes", "queries",
        "episode_fingerprint", "roc_auc", "accuracy", "f1",
    ]
    with (output_root / "classification_long.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=long_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "sources": ""})

    random_df = pd.DataFrame(rows)
    trained_means = trained.groupby(["architecture", "dataset"]).roc_auc.mean()
    target_rows = []
    for row in random_df.itertuples(index=False):
        trained_mean = float(trained_means.loc[(row.architecture, row.dataset)])
        trained_cells = trained[
            (trained.architecture == row.architecture) & (trained.dataset == row.dataset)
        ].roc_auc
        target_rows.append(
            {
                "architecture": row.architecture,
                "dataset": row.dataset,
                "random_init_roc_auc": float(row.roc_auc),
                "update100_mean_roc_auc": trained_mean,
                "mean_delta": trained_mean - float(row.roc_auc),
                "update100_cells_above_random_fraction": float(
                    (trained_cells > float(row.roc_auc)).mean()
                ),
            }
        )
    target_df = pd.DataFrame(target_rows)
    target_df.to_csv(output_root / "target_summary.csv", index=False)

    summary = {
        "protocol": "one_seed_fixed_episode_random_initialization_control",
        "seed": 0,
        "checkpoint_step": 0,
        "rows": len(rows),
        "targets": list(TARGETS),
        "episode_fingerprints": {
            target: next(row["episode_fingerprint"] for row in rows if row["dataset"] == target)
            for target in TARGETS
        },
        "input_sha256": {
            architecture: hashlib.sha256(path.read_bytes()).hexdigest()
            for architecture, path in input_paths.items()
        },
        "mean_roc_auc": {
            architecture: float(
                random_df[random_df.architecture == architecture].roc_auc.mean()
            )
            for architecture in ARCHITECTURES
        },
        "update100_mean_roc_auc": {
            architecture: float(
                trained[trained.architecture == architecture].roc_auc.mean()
            )
            for architecture in ARCHITECTURES
        },
        "mean_delta_update100_minus_random": {
            architecture: float(
                target_df[target_df.architecture == architecture].mean_delta.mean()
            )
            for architecture in ARCHITECTURES
        },
        "update100_cells_above_random_fraction": {
            architecture: float(
                target_df[
                    target_df.architecture == architecture
                ].update100_cells_above_random_fraction.mean()
            )
            for architecture in ARCHITECTURES
        },
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
