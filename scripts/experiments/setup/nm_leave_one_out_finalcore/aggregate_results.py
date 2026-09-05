#!/usr/bin/env python3
"""Strictly validate and aggregate nine held-out LOO NM evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
FINAL_CORE = HERE.parent / "final_core"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FINAL_CORE))

from auc_contract import METRIC_CONTRACT  # noqa: E402
from loo_plan import (  # noqa: E402
    CHECKPOINT_STEP,
    EPISODE_COUNT,
    PROTOCOL,
    SEEDS,
    SOURCES,
    checkpoint_path,
    physical_jobs,
)


METRICS = ("accuracy", "f1_macro", "roc_auc_ovr_macro")


def atomic_table(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
    *,
    delimiter: str = "\t",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def expected_paths(results_root: Path) -> set[Path]:
    return {
        results_root / f"seed_{job.seed}" / job.model.model_id / f"{job.model.heldout}.json"
        for job in physical_jobs()
    }


def load_and_validate(
    results_root: Path,
    training_run_dir: Path,
    expected_batch_size: int,
) -> list[dict[str, Any]]:
    expected = expected_paths(results_root)
    actual = set(results_root.glob("seed_*/*/*.json"))
    if actual != expected:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        raise ValueError(
            f"result path mismatch: missing={len(missing)} extra={len(extra)}; "
            f"first_missing={missing[:1]} first_extra={extra[:1]}"
        )
    job_by_key = {job.key: job for job in physical_jobs()}
    rows = []
    for path in sorted(expected):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(path.parents[1].name.removeprefix("seed_"))
        model_id = path.parent.name
        target = path.stem
        job = job_by_key[(seed, model_id)]
        checks = {
            "protocol": PROTOCOL,
            "metric_contract": METRIC_CONTRACT,
            "checkpoint_step": CHECKPOINT_STEP,
            "split": "test",
            "edge_view": "static_train",
            "target_edge_view": "static_test",
            "batch_size": expected_batch_size,
            "batch_count": EPISODE_COUNT // expected_batch_size,
            "episode_count": EPISODE_COUNT,
            "seed": seed,
            "model_id": model_id,
            "target": job.model.heldout,
            "sources": list(job.model.sources),
            "checkpoint": str(checkpoint_path(training_run_dir, job, "")),
        }
        for key, wanted in checks.items():
            if payload.get(key) != wanted:
                raise ValueError(f"{path}: {key} expected {wanted!r}, got {payload.get(key)!r}")
        if target != job.model.heldout or target in job.model.sources:
            raise ValueError(f"{path}: target is not the model's excluded source")
        for key in ("score", "score_std", "loss", "aux_loss", *METRICS):
            if not math.isfinite(float(payload.get(key, float("nan")))):
                raise ValueError(f"{path}: missing or non-finite {key}")
        if not math.isclose(float(payload["score"]), float(payload["accuracy"]), abs_tol=1e-12):
            raise ValueError(f"{path}: score does not equal accuracy")
        for key in ("episode_plan_fingerprint", "observed_episode_fingerprint"):
            if len(payload.get(key, "")) != 64:
                raise ValueError(f"{path}: invalid {key}")
        rows.append(payload)
    if len(rows) != 9:
        raise AssertionError(f"expected 9 held-out cells, got {len(rows)}")
    return rows


def aggregate(
    results_root: Path,
    output_root: Path,
    training_run_dir: Path,
    expected_batch_size: int,
) -> None:
    payloads = load_and_validate(results_root, training_run_dir, expected_batch_size)
    rows = []
    for payload in payloads:
        rows.append({
            "seed": payload["seed"],
            "model_id": payload["model_id"],
            "heldout": payload["target"],
            "training_sources": ",".join(payload["sources"]),
            **{metric: payload[metric] for metric in METRICS},
            "checkpoint_step": payload["checkpoint_step"],
            "episode_count": payload["episode_count"],
            "episode_plan_fingerprint": payload["episode_plan_fingerprint"],
            "observed_episode_fingerprint": payload["observed_episode_fingerprint"],
        })
    order = {source: index for index, source in enumerate(SOURCES)}
    rows.sort(key=lambda row: order[row["heldout"]])
    fields = list(rows[0])
    atomic_table(output_root / "loo_heldout_metrics.tsv", fields, rows)
    atomic_table(output_root / "loo_heldout_metrics.csv", fields, rows, delimiter=",")

    completeness = {
        "protocol": PROTOCOL,
        "metric_contract": METRIC_CONTRACT,
        "checkpoint_step": CHECKPOINT_STEP,
        "batch_size": expected_batch_size,
        "episode_count_per_cell": EPISODE_COUNT,
        "training_seeds": list(SEEDS),
        "sources": list(SOURCES),
        "loo_models": 9,
        "heldout_test_cells": len(rows),
        "metrics": list(METRICS),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "completeness.json").write_text(
        json.dumps(completeness, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--training-run-dir", required=True, type=Path)
    parser.add_argument("--expected-batch-size", default=32, type=int)
    args = parser.parse_args()
    if args.expected_batch_size <= 0 or EPISODE_COUNT % args.expected_batch_size:
        parser.error("expected-batch-size must divide 512")
    aggregate(args.results_root, args.output_root, args.training_run_dir, args.expected_batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
