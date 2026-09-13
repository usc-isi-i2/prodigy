#!/usr/bin/env python3
"""Validate and aggregate the frozen 729-cell Stage-A pilot grid."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
FINAL_CORE = HERE.parent / "final_core"
sys.path.insert(0, str(FINAL_CORE))
sys.path.insert(0, str(HERE))

from aggregate_fixed_test import atomic_table, write_matrix  # noqa: E402
from auc_contract import METRIC_CONTRACT  # noqa: E402
from build_manifest import (  # noqa: E402
    EPISODES,
    PILOT_STEPS,
    PROTOCOL_ID,
    SEEDS,
    SOURCES,
)


METRICS = ("accuracy", "f1_macro", "roc_auc_ovr_macro")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 729 or len({row["cell_id"] for row in rows}) != 729:
        raise ValueError("Stage-A manifest must contain 729 unique cells")
    return rows


def expected_path(results_root: Path, row: dict[str, str]) -> Path:
    return (
        results_root
        / f"step_{row['checkpoint_step']}"
        / f"seed_{row['training_seed']}"
        / row["model_id"]
        / f"{row['target']}.json"
    )


def validate_payload(
    path: Path, payload: dict[str, Any], manifest_row: dict[str, str], batch_size: int
) -> None:
    checks: dict[str, object] = {
        "protocol": PROTOCOL_ID,
        "metric_contract": METRIC_CONTRACT,
        "model_id": manifest_row["model_id"],
        "seed": int(manifest_row["training_seed"]),
        "target": manifest_row["target"],
        "checkpoint_step": int(manifest_row["checkpoint_step"]),
        "checkpoint": manifest_row["checkpoint"],
        "split": "val",
        "edge_view": "static_train",
        "target_edge_view": "static_validation",
        "batch_size": batch_size,
        "batch_count": EPISODES // batch_size,
        "episode_count": EPISODES,
    }
    for key, wanted in checks.items():
        if payload.get(key) != wanted:
            raise ValueError(
                f"{path}: {key} expected {wanted!r}, got {payload.get(key)!r}"
            )
    if payload.get("sources") != [manifest_row["sources"]]:
        raise ValueError(f"{path}: source identity disagrees with manifest")
    if Path(payload["checkpoint"]).name != (
        f"state_dict_{manifest_row['checkpoint_step']}.ckpt"
    ):
        raise ValueError(f"{path}: checkpoint filename disagrees with step")
    for key in ("score", "score_std", "loss", "aux_loss", *METRICS):
        if not math.isfinite(float(payload.get(key, float("nan")))):
            raise ValueError(f"{path}: missing or non-finite {key}")
    if not math.isclose(float(payload["score"]), float(payload["accuracy"]), abs_tol=1e-12):
        raise ValueError(f"{path}: score does not equal accuracy")
    for key in ("episode_plan_fingerprint", "observed_episode_fingerprint"):
        if len(payload.get(key, "")) != 64:
            raise ValueError(f"{path}: invalid {key}")


def load_and_validate(
    results_root: Path, manifest_path: Path, batch_size: int
) -> list[dict[str, Any]]:
    manifest_rows = read_manifest(manifest_path)
    expected = {expected_path(results_root, row): row for row in manifest_rows}
    actual = set(results_root.glob("step_*/seed_*/*/*.json"))
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise ValueError(
            f"result path mismatch: missing={len(missing)} extra={len(extra)}; "
            f"first_missing={missing[:1]} first_extra={extra[:1]}"
        )
    payloads = []
    for path, row in sorted(expected.items()):
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_payload(path, payload, row, batch_size)
        payloads.append(payload)

    for target in SOURCES:
        target_rows = [row for row in payloads if row["target"] == target]
        if len(target_rows) != len(PILOT_STEPS) * len(SEEDS) * len(SOURCES):
            raise ValueError(f"target {target} has an incomplete result panel")
        for key in ("episode_plan_fingerprint", "observed_episode_fingerprint"):
            if len({row[key] for row in target_rows}) != 1:
                raise ValueError(f"target {target} disagrees on {key}")
    return payloads


def aggregate(
    results_root: Path, manifest_path: Path, output_root: Path, batch_size: int
) -> None:
    payloads = load_and_validate(results_root, manifest_path, batch_size)
    rows = []
    by_key: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    for payload in payloads:
        source = payload["model_id"].removeprefix("ss_")
        key = (
            int(payload["checkpoint_step"]),
            int(payload["seed"]),
            source,
            payload["target"],
        )
        if key in by_key:
            raise AssertionError(f"duplicate result cell {key}")
        by_key[key] = payload
        rows.append(
            {
                "checkpoint_step": key[0],
                "seed": key[1],
                "source": source,
                "target": key[3],
                **{metric: payload[metric] for metric in METRICS},
                "episode_count": payload["episode_count"],
                "episode_plan_fingerprint": payload["episode_plan_fingerprint"],
                "observed_episode_fingerprint": payload[
                    "observed_episode_fingerprint"
                ],
            }
        )
    rows.sort(key=lambda row: (
        row["checkpoint_step"], row["seed"], row["source"], row["target"]
    ))
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_table(output_root / "pilot_metrics_long.tsv", list(rows[0]), rows)

    for step in PILOT_STEPS:
        for metric in METRICS:
            seed_values = {}
            for seed in SEEDS:
                values = {
                    (source, target): float(by_key[(step, seed, source, target)][metric])
                    for source in SOURCES
                    for target in SOURCES
                }
                seed_values[seed] = values
                write_matrix(
                    output_root / f"pilot_{metric}_step{step}_seed{seed}.csv",
                    list(SOURCES),
                    values,
                )
            means = {
                (source, target): statistics.mean(
                    seed_values[seed][(source, target)] for seed in SEEDS
                )
                for source in SOURCES
                for target in SOURCES
            }
            stds = {
                (source, target): statistics.stdev(
                    seed_values[seed][(source, target)] for seed in SEEDS
                )
                for source in SOURCES
                for target in SOURCES
            }
            write_matrix(
                output_root / f"pilot_{metric}_step{step}_three_seed_mean.csv",
                list(SOURCES),
                means,
            )
            write_matrix(
                output_root / f"pilot_{metric}_step{step}_three_seed_sample_std.csv",
                list(SOURCES),
                stds,
            )

    completeness = {
        "protocol": PROTOCOL_ID,
        "metric_contract": METRIC_CONTRACT,
        "checkpoint_steps": list(PILOT_STEPS),
        "training_seeds": list(SEEDS),
        "sources": list(SOURCES),
        "targets": list(SOURCES),
        "split": "val",
        "edge_view": "static_train",
        "target_edge_view": "static_validation",
        "batch_size": batch_size,
        "episode_count_per_cell": EPISODES,
        "cells": len(rows),
        "metrics": list(METRICS),
    }
    (output_root / "completeness.json").write_text(
        json.dumps(completeness, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", default=32, type=int)
    args = parser.parse_args()
    if args.batch_size <= 0 or EPISODES % args.batch_size:
        parser.error("batch-size must be a positive divisor of 512")
    aggregate(args.results_root, args.manifest, args.output_root, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
