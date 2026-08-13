#!/usr/bin/env python3
"""Validate and aggregate the 243-cell final-core specialist AUC grid."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from aggregate_fixed_test import atomic_table, write_matrix  # noqa: E402
from auc_contract import METRIC_CONTRACT  # noqa: E402
from core_plan import SOURCES  # noqa: E402
from fixed_test_plan import CHECKPOINT_STEP, EPISODE_COUNT, PROTOCOL, SEEDS, physical_jobs  # noqa: E402


METRICS = ("accuracy", "f1_macro", "roc_auc_ovr_macro")


def specialist_jobs():
    jobs = [job for job in physical_jobs() if job.model.model_id.startswith("ss_")]
    if len(jobs) != 27:
        raise AssertionError(f"expected 27 specialist checkpoints, got {len(jobs)}")
    return jobs


def expected_result_paths(results_root: Path) -> set[Path]:
    return {
        results_root / f"seed_{job.seed}" / job.model.model_id / f"{target}.json"
        for job in specialist_jobs()
        for target in SOURCES
    }


def load_and_validate(results_root: Path, expected_batch_size: int) -> list[dict[str, Any]]:
    expected = expected_result_paths(results_root)
    actual = set(results_root.glob("seed_*/*/*.json"))
    if actual != expected:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        raise ValueError(
            f"result path mismatch: missing={len(missing)} extra={len(extra)}; "
            f"first_missing={missing[:1]} first_extra={extra[:1]}"
        )
    rows = []
    for path in sorted(expected):
        payload = json.loads(path.read_text(encoding="utf-8"))
        path_seed = int(path.parents[1].name.removeprefix("seed_"))
        path_model = path.parent.name
        path_target = path.stem
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
            "seed": path_seed,
            "model_id": path_model,
            "target": path_target,
        }
        for key, wanted in checks.items():
            if payload.get(key) != wanted:
                raise ValueError(f"{path}: {key} expected {wanted!r}, got {payload.get(key)!r}")
        if not payload.get("model_id", "").startswith("ss_"):
            raise ValueError(f"{path}: non-specialist model")
        if Path(payload["checkpoint"]).name != f"state_dict_{CHECKPOINT_STEP}.ckpt":
            raise ValueError(f"{path}: wrong checkpoint")
        for key in ("score", "score_std", "loss", "aux_loss", *METRICS):
            if not math.isfinite(float(payload.get(key, float("nan")))):
                raise ValueError(f"{path}: missing or non-finite {key}")
        if not math.isclose(float(payload["score"]), float(payload["accuracy"]), abs_tol=1e-12):
            raise ValueError(f"{path}: score does not equal accuracy")
        for key in ("episode_plan_fingerprint", "observed_episode_fingerprint"):
            if len(payload.get(key, "")) != 64:
                raise ValueError(f"{path}: invalid {key}")
        rows.append(payload)
    if len(rows) != 243:
        raise AssertionError(f"expected 243 cells, got {len(rows)}")
    for target in SOURCES:
        target_rows = [row for row in rows if row["target"] == target]
        for key in ("episode_plan_fingerprint", "observed_episode_fingerprint"):
            if len({row[key] for row in target_rows}) != 1:
                raise ValueError(f"target {target} disagrees on {key}")
    return rows


def aggregate(results_root: Path, output_root: Path, expected_batch_size: int) -> None:
    payloads = load_and_validate(results_root, expected_batch_size)
    rows = []
    by_key = {}
    for payload in payloads:
        source = payload["model_id"].removeprefix("ss_")
        key = (int(payload["seed"]), source, payload["target"])
        if key in by_key:
            raise AssertionError(f"duplicate cell {key}")
        by_key[key] = payload
        rows.append({
            "seed": payload["seed"],
            "source": source,
            "target": payload["target"],
            **{metric: payload[metric] for metric in METRICS},
            "checkpoint_step": payload["checkpoint_step"],
            "episode_count": payload["episode_count"],
            "episode_plan_fingerprint": payload["episode_plan_fingerprint"],
            "observed_episode_fingerprint": payload["observed_episode_fingerprint"],
        })
    rows.sort(key=lambda row: (row["seed"], row["source"], row["target"]))
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_table(output_root / "single_source_metrics_long.tsv", list(rows[0]), rows)

    for metric in METRICS:
        by_seed = {}
        for seed in SEEDS:
            values = {
                (source, target): float(by_key[(seed, source, target)][metric])
                for source in SOURCES for target in SOURCES
            }
            if len(values) != 81:
                raise AssertionError(f"seed {seed} {metric} matrix is incomplete")
            by_seed[seed] = values
            write_matrix(output_root / f"single_source_{metric}_seed_{seed}.csv", list(SOURCES), values)
        mean_values, std_values = {}, {}
        for source in SOURCES:
            for target in SOURCES:
                values = [by_seed[seed][(source, target)] for seed in SEEDS]
                mean_values[(source, target)] = statistics.mean(values)
                std_values[(source, target)] = statistics.stdev(values)
        write_matrix(
            output_root / f"single_source_{metric}_three_seed_mean.csv",
            list(SOURCES), mean_values,
        )
        write_matrix(
            output_root / f"single_source_{metric}_three_seed_sample_std.csv",
            list(SOURCES), std_values,
        )

    completeness = {
        "protocol": PROTOCOL,
        "metric_contract": METRIC_CONTRACT,
        "checkpoint_step": CHECKPOINT_STEP,
        "batch_size": expected_batch_size,
        "batch_count": EPISODE_COUNT // expected_batch_size,
        "episode_count_per_cell": EPISODE_COUNT,
        "training_seeds": list(SEEDS),
        "sources": list(SOURCES),
        "targets": list(SOURCES),
        "specialist_cells": len(rows),
        "metrics": list(METRICS),
    }
    (output_root / "completeness.json").write_text(
        json.dumps(completeness, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-batch-size", required=True, type=int)
    args = parser.parse_args()
    if args.expected_batch_size <= 0 or EPISODE_COUNT % args.expected_batch_size:
        parser.error("expected-batch-size must divide 512")
    aggregate(args.results_root, args.output_root, args.expected_batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
