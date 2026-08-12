#!/usr/bin/env python3
"""Validate the 18 clean specialist cells and compare with frozen fixed-test scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from protocol import EPISODE_COUNT, PROTOCOL, TARGETS


SPECIALISTS = tuple(f"ss_{target}" for target in TARGETS)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--original-results-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    plan_by_target: dict[str, set[str]] = {target: set() for target in TARGETS}
    observed_by_target: dict[str, set[str]] = {target: set() for target in TARGETS}
    context_by_target: dict[str, set[tuple[int, int, int]]] = {target: set() for target in TARGETS}
    for seed in (0, 1, 2):
        for model_id in SPECIALISTS:
            source = model_id.removeprefix("ss_")
            for target in TARGETS:
                if source == target:
                    continue
                clean_path = args.results_root / f"seed_{seed}" / model_id / f"{target}.json"
                original_path = args.original_results_root / f"seed_{seed}" / model_id / f"{target}.json"
                if not clean_path.is_file() or not original_path.is_file():
                    raise FileNotFoundError(f"missing clean/original cell for {(seed, source, target)}")
                clean = json.loads(clean_path.read_text())
                original = json.loads(original_path.read_text())
                checks = {
                    "protocol": PROTOCOL,
                    "seed": seed,
                    "model_id": model_id,
                    "target": target,
                    "episode_count": EPISODE_COUNT,
                    "exclusion_level": "episode_centers",
                }
                for field, wanted in checks.items():
                    if clean.get(field) != wanted:
                        raise ValueError(f"{clean_path}: {field} mismatch")
                if clean["unfiltered_prefix_plan_fingerprint"] != original["episode_plan_fingerprint"]:
                    raise ValueError(
                        f"{clean_path}: unfiltered prefix does not reproduce frozen original plan"
                    )
                for field in ("score", "score_std", "loss", "aux_loss"):
                    if not math.isfinite(float(clean[field])):
                        raise ValueError(f"{clean_path}: non-finite {field}")
                plan_by_target[target].add(clean["episode_plan_fingerprint"])
                observed_by_target[target].add(clean["observed_episode_fingerprint"])
                context_by_target[target].add((
                    int(clean["sampled_context_node_occurrences"]),
                    int(clean["sampled_context_overlap_occurrences"]),
                    int(clean["sampled_context_unique_overlap_nodes"]),
                ))
                rows.append({
                    "seed": seed,
                    "source": source,
                    "target": target,
                    "original_score": float(original["score"]),
                    "center_clean_score": float(clean["score"]),
                    "delta_clean_minus_original": float(clean["score"]) - float(original["score"]),
                    "excluded_target_nodes": int(clean["excluded_node_count"]),
                    "target_graph_nodes": int(clean["target_graph_nodes"]),
                    "sampled_context_node_occurrences": int(clean["sampled_context_node_occurrences"]),
                    "sampled_context_overlap_occurrences": int(clean["sampled_context_overlap_occurrences"]),
                    "sampled_context_unique_overlap_nodes": int(clean["sampled_context_unique_overlap_nodes"]),
                    "episode_plan_fingerprint": clean["episode_plan_fingerprint"],
                    "observed_episode_fingerprint": clean["observed_episode_fingerprint"],
                })
    if len(rows) != 18:
        raise AssertionError(f"expected 18 off-diagonal cells, got {len(rows)}")
    for target in TARGETS:
        if len(plan_by_target[target]) != 1 or len(observed_by_target[target]) != 1:
            raise ValueError(f"{target}: clean episodes differ across checkpoints")
        if len(context_by_target[target]) != 1:
            raise ValueError(f"{target}: sampled context differs across checkpoints")
    rows.sort(key=lambda row: (row["source"], row["target"], row["seed"]))
    write_tsv(args.output_root / "paired_cells.tsv", rows)
    pair_summaries = []
    for source in TARGETS:
        for target in TARGETS:
            if source == target:
                continue
            selected = [row for row in rows if row["source"] == source and row["target"] == target]
            pair_summaries.append({
                "source": source,
                "target": target,
                "original_mean": statistics.mean(row["original_score"] for row in selected),
                "center_clean_mean": statistics.mean(row["center_clean_score"] for row in selected),
                "delta_mean": statistics.mean(row["delta_clean_minus_original"] for row in selected),
                "seed_deltas": [row["delta_clean_minus_original"] for row in selected],
            })
    context_by_target_summary = {}
    for target in TARGETS:
        row = next(item for item in rows if item["target"] == target)
        total = int(row["sampled_context_node_occurrences"])
        overlap = int(row["sampled_context_overlap_occurrences"])
        context_by_target_summary[target] = {
            "sampled_node_occurrences": total,
            "overlap_occurrences": overlap,
            "overlap_occurrence_fraction": overlap / total,
            "unique_overlap_nodes": int(row["sampled_context_unique_overlap_nodes"]),
        }
    payload = {
        "protocol": PROTOCOL,
        "cells": 18,
        "seeds": 3,
        "directions": 6,
        "original_mean": statistics.mean(row["original_score"] for row in rows),
        "center_clean_mean": statistics.mean(row["center_clean_score"] for row in rows),
        "delta_mean": statistics.mean(row["delta_clean_minus_original"] for row in rows),
        "directions_improved": sum(item["delta_mean"] > 0 for item in pair_summaries),
        "pairs": pair_summaries,
        "residual_sampled_context_overlap": context_by_target_summary,
        "episode_plan_fingerprints": {target: next(iter(values)) for target, values in plan_by_target.items()},
        "observed_episode_fingerprints": {target: next(iter(values)) for target, values in observed_by_target.items()},
        "paired_cells_sha256": hashlib.sha256((args.output_root / "paired_cells.tsv").read_bytes()).hexdigest(),
    }
    atomic_json(args.output_root / "summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
