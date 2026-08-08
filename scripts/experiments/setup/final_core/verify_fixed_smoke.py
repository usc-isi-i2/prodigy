#!/usr/bin/env python3
"""Verify the concurrent full-batch fixed-test smoke run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from auc_contract import METRIC_CONTRACT  # noqa: E402
from core_plan import SOURCES  # noqa: E402
from fixed_test_plan import CHECKPOINT_STEP, EPISODE_COUNT, PROTOCOL  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--ready-dir", required=True, type=Path)
    parser.add_argument("--expected-workers", default=8, type=int)
    parser.add_argument("--expected-cells", default=10, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--max-summed-vram-gib-per-gpu", default=70.0, type=float)
    parser.add_argument("--min-host-reserve-gib", default=256.0, type=float)
    args = parser.parse_args()
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.results_root.glob("seed_*/*/*.json"))
    ]
    if len(payloads) != args.expected_cells:
        raise AssertionError(f"expected {args.expected_cells} smoke cells, got {len(payloads)}")
    ready = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.ready_dir.glob("worker_*.json"))
    ]
    if len(ready) != args.expected_workers:
        raise AssertionError(f"expected {args.expected_workers} ready workers, got {len(ready)}")
    if {row["worker_index"] for row in ready} != set(range(args.expected_workers)):
        raise AssertionError("smoke ready-worker ids are incomplete")
    expected_batches = EPISODE_COUNT // args.batch_size
    for row in payloads:
        expected = {
            "protocol": PROTOCOL,
            "checkpoint_step": CHECKPOINT_STEP,
            "batch_size": args.batch_size,
            "batch_count": expected_batches,
            "episode_count": EPISODE_COUNT,
            "edge_view": "static_train",
            "target_edge_view": "static_test",
            "batch_replay_mode": "materialized_cpu_clone_v1",
            "metric_contract": METRIC_CONTRACT,
        }
        for field, value in expected.items():
            if row.get(field) != value:
                raise AssertionError(f"smoke {field}: expected {value!r}, got {row.get(field)!r}")
        for field in ("accuracy", "f1_macro", "roc_auc_ovr_macro"):
            value = float(row.get(field, float("nan")))
            if not 0.0 <= value <= 1.0:
                raise AssertionError(f"smoke {field} is invalid: {value}")
        if float(row["cache_min_mem_available_gib"]) < args.min_host_reserve_gib:
            raise MemoryError(
                f"worker {row['worker_index']} cache left only "
                f"{row['cache_min_mem_available_gib']:.1f} GiB available"
            )
    if {row["target"] for row in payloads} != set(SOURCES):
        raise AssertionError("smoke run did not cover all nine targets")
    for target in SOURCES:
        target_rows = [row for row in payloads if row["target"] == target]
        for field in ("episode_plan_fingerprint", "observed_episode_fingerprint"):
            if len({row[field] for row in target_rows}) != 1:
                raise AssertionError(f"smoke target {target} disagrees on {field}")
    repeated_across_workers = [
        target
        for target in SOURCES
        if len(
            {
                row["worker_index"]
                for row in payloads
                if row["target"] == target
            }
        )
        > 1
    ]
    if repeated_across_workers != ["ukr_rus"]:
        raise AssertionError(
            "expected cross-worker ukr_rus repeat, got "
            f"{repeated_across_workers}"
        )
    per_worker_peak = {}
    for row in payloads:
        worker = int(row["worker_index"])
        per_worker_peak[worker] = max(
            per_worker_peak.get(worker, 0.0), float(row["peak_cuda_allocated_gib"])
        )
    for gpu in range(4):
        summed = per_worker_peak.get(gpu * 2, 0.0) + per_worker_peak.get(gpu * 2 + 1, 0.0)
        if summed > args.max_summed_vram_gib_per_gpu:
            raise MemoryError(
                f"GPU {gpu} conservative summed worker peak is {summed:.1f} GiB; "
                f"limit is {args.max_summed_vram_gib_per_gpu:.1f} GiB"
            )
    print(
        json.dumps(
            {
                "status": "ok",
                "batch_size": args.batch_size,
                "batch_count": expected_batches,
                "cells": len(payloads),
                "workers": len(ready),
                "min_mem_available_at_barrier_gib": min(row["mem_available_gib"] for row in ready),
                "max_worker_rss_at_barrier_gib": max(row["max_rss_gib"] for row in ready),
                "max_worker_rss_after_cache_gib": max(row["max_rss_gib"] for row in payloads),
                "min_mem_available_during_cache_gib": min(
                    row["cache_min_mem_available_gib"] for row in payloads
                ),
                "mean_cache_materialization_seconds": sum(
                    row["cache_materialization_seconds"] for row in payloads
                ) / len(payloads),
                "mean_replay_cell_seconds": sum(row["elapsed_seconds"] for row in payloads)
                / len(payloads),
                "per_worker_peak_cuda_gib": per_worker_peak,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
