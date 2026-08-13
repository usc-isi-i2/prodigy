#!/usr/bin/env python3
"""Strictly validate and assemble the 837-cell final-core fixed-test sweep."""

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
sys.path.insert(0, str(HERE))

from core_plan import ORDERS, SOURCES, build_models  # noqa: E402
from fixed_test_plan import (  # noqa: E402
    CHECKPOINT_STEP,
    EPISODE_COUNT,
    PROTOCOL,
    SEEDS,
    expected_counts,
    model_for_ladder,
    physical_jobs,
)


def atomic_table(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
    *,
    delimiter: str = "\t",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_matrix(path: Path, row_names: list[str], values: dict[tuple[str, str], float]) -> None:
    rows = []
    for row_name in row_names:
        row: dict[str, Any] = {"model_source": row_name}
        for target in SOURCES:
            row[target] = values[(row_name, target)]
        rows.append(row)
    atomic_table(path, ["model_source", *SOURCES], rows, delimiter=",")


def expected_result_paths(results_root: Path) -> set[Path]:
    return {
        results_root / f"seed_{job.seed}" / job.model.model_id / f"{target}.json"
        for job in physical_jobs()
        for target in SOURCES
    }


def load_and_validate(results_root: Path, *, expected_batch_size: int) -> list[dict[str, Any]]:
    expected = expected_result_paths(results_root)
    actual = set(results_root.glob("seed_*/*/*.json"))
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"result path mismatch: missing={len(missing)} extra={len(extra)}; "
            f"first_missing={missing[:1]} first_extra={extra[:1]}"
        )
    expected_batch_count = EPISODE_COUNT // expected_batch_size
    rows = []
    keys = set()
    for path in sorted(expected):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (int(payload.get("seed", -1)), payload.get("model_id"), payload.get("target"))
        if key in keys:
            raise ValueError(f"duplicate physical key {key}")
        keys.add(key)
        checks = {
            "protocol": PROTOCOL,
            "checkpoint_step": CHECKPOINT_STEP,
            "split": "test",
            "edge_view": "static_train",
            "target_edge_view": "static_test",
            "batch_size": expected_batch_size,
            "batch_count": expected_batch_count,
            "episode_count": EPISODE_COUNT,
        }
        for field, wanted in checks.items():
            if payload.get(field) != wanted:
                raise ValueError(
                    f"{path}: {field} expected {wanted!r}, got {payload.get(field)!r}"
                )
        if Path(payload["checkpoint"]).name != f"state_dict_{CHECKPOINT_STEP}.ckpt":
            raise ValueError(f"{path}: wrong checkpoint {payload['checkpoint']}")
        forbidden = {
            "validation_results", "selected_checkpoint", "selected_checkpoint_step",
            "selection_created_utc", "selection_rule",
        }
        present = sorted(forbidden & payload.keys())
        if present:
            raise ValueError(f"{path}: validation/selection fields are forbidden: {present}")
        for field in ("score", "score_std", "loss", "aux_loss"):
            if not math.isfinite(float(payload[field])):
                raise ValueError(f"{path}: non-finite {field}")
        for field in ("episode_plan_fingerprint", "observed_episode_fingerprint"):
            value = payload.get(field, "")
            if len(value) != 64:
                raise ValueError(f"{path}: invalid {field}")
        rows.append(payload)
    if len(rows) != expected_counts()["union_cells"] or len(keys) != len(rows):
        raise AssertionError("combined physical result grid is not exactly 837 cells")
    for target in SOURCES:
        target_rows = [row for row in rows if row["target"] == target]
        for field in ("episode_plan_fingerprint", "observed_episode_fingerprint"):
            fingerprints = {row[field] for row in target_rows}
            if len(fingerprints) != 1:
                raise ValueError(
                    f"target {target} has {len(fingerprints)} distinct {field} values"
                )
    return rows


def physical_row(payload: dict[str, Any], model_by_id) -> dict[str, Any]:
    model = model_by_id[payload["model_id"]]
    return {
        "seed": payload["seed"],
        "model_id": payload["model_id"],
        "n_sources": len(model.sources),
        "sources": ",".join(model.sources),
        "target": payload["target"],
        "checkpoint_step": payload["checkpoint_step"],
        "score": payload["score"],
        "score_std_across_batches": payload["score_std"],
        "loss": payload["loss"],
        "batch_size": payload["batch_size"],
        "batch_count": payload["batch_count"],
        "episode_count": payload["episode_count"],
        "episode_plan_fingerprint": payload["episode_plan_fingerprint"],
        "observed_episode_fingerprint": payload["observed_episode_fingerprint"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }


def aggregate(results_root: Path, output_root: Path, *, expected_batch_size: int) -> None:
    counts = expected_counts()
    payloads = load_and_validate(results_root, expected_batch_size=expected_batch_size)
    models = build_models()
    model_by_id = {model.model_id: model for model in models}
    by_key = {
        (int(row["seed"]), row["model_id"], row["target"]): row for row in payloads
    }

    combined_rows = [physical_row(row, model_by_id) for row in payloads]
    combined_rows.sort(key=lambda row: (row["seed"], row["model_id"], row["target"]))
    if len(combined_rows) != counts["union_cells"]:
        raise AssertionError("combined table must have 837 rows")

    specialist_ids = {f"ss_{source}" for source in SOURCES}
    matrix_rows = [row for row in combined_rows if row["model_id"] in specialist_ids]
    matrix_keys = {(row["seed"], row["model_id"], row["target"]) for row in matrix_rows}
    if len(matrix_rows) != counts["matrix_cells"] or len(matrix_keys) != len(matrix_rows):
        raise AssertionError("specialist matrix must contain 243 unique cells")

    ladder_ids = {
        model_for_ladder(order, rung).model_id
        for order in ORDERS
        for rung in range(1, 10)
    }
    ladder_physical = [row for row in combined_rows if row["model_id"] in ladder_ids]
    ladder_physical_keys = {
        (row["seed"], row["model_id"], row["target"]) for row in ladder_physical
    }
    if (
        len(ladder_physical) != counts["ladder_physical_cells"]
        or len(ladder_physical_keys) != len(ladder_physical)
    ):
        raise AssertionError("ladder physical table must contain 675 unique cells")

    ladder_rows = []
    overlap_rows = []
    ladder_alias_keys = set()
    for seed in SEEDS:
        for order in ORDERS:
            for rung in range(1, 10):
                model = model_for_ladder(order, rung)
                for target in SOURCES:
                    payload = by_key[(seed, model.model_id, target)]
                    key = (seed, order, rung, target)
                    if key in ladder_alias_keys:
                        raise AssertionError(f"duplicate ladder alias key {key}")
                    ladder_alias_keys.add(key)
                    row = {
                        "seed": seed,
                        "order": order,
                        "rung": rung,
                        "model_id": model.model_id,
                        "sources": ",".join(model.sources),
                        "target": target,
                        "score": payload["score"],
                        "checkpoint_step": payload["checkpoint_step"],
                        "episode_count": payload["episode_count"],
                        "episode_plan_fingerprint": payload["episode_plan_fingerprint"],
                        "observed_episode_fingerprint": payload["observed_episode_fingerprint"],
                    }
                    ladder_rows.append(row)
                    if rung == 1:
                        matrix_payload = by_key[(seed, f"ss_{model.sources[0]}", target)]
                        if matrix_payload is not payload:
                            raise AssertionError("rung-1/matrix overlap was not physically reused")
                        overlap_rows.append(row)
    if len(ladder_rows) != counts["ladder_reported_rows"]:
        raise AssertionError("alias-expanded ladder must contain exactly 729 rows")
    if len(overlap_rows) != counts["overlap_cells"]:
        raise AssertionError("rung-1 overlap must contain exactly 81 rows")

    matrix_by_seed: dict[int, dict[tuple[str, str], float]] = {}
    for seed in SEEDS:
        values = {}
        for source in SOURCES:
            for target in SOURCES:
                values[(source, target)] = float(
                    by_key[(seed, f"ss_{source}", target)]["score"]
                )
        if len(values) != 81:
            raise AssertionError(f"seed {seed} matrix is not 9x9")
        matrix_by_seed[seed] = values
        write_matrix(output_root / f"single_source_matrix_seed_{seed}.csv", list(SOURCES), values)

    mean_values = {}
    std_values = {}
    for source in SOURCES:
        for target in SOURCES:
            scores = [matrix_by_seed[seed][(source, target)] for seed in SEEDS]
            mean_values[(source, target)] = statistics.mean(scores)
            std_values[(source, target)] = statistics.stdev(scores)
    write_matrix(output_root / "single_source_matrix_three_seed_mean.csv", list(SOURCES), mean_values)
    write_matrix(output_root / "single_source_matrix_three_seed_sample_std.csv", list(SOURCES), std_values)

    fingerprint_rows = []
    for target in SOURCES:
        target_payloads = [row for row in payloads if row["target"] == target]
        fingerprint_rows.append({
            "target": target,
            "cell_count": len(target_payloads),
            "episode_count_per_cell": EPISODE_COUNT,
            "episode_plan_fingerprint": target_payloads[0]["episode_plan_fingerprint"],
            "observed_episode_fingerprint": target_payloads[0]["observed_episode_fingerprint"],
        })

    atomic_table(output_root / "combined_physical_cells.tsv", list(combined_rows[0]), combined_rows)
    atomic_table(output_root / "single_source_matrix_long.tsv", list(matrix_rows[0]), matrix_rows)
    atomic_table(output_root / "ladder_physical_cells.tsv", list(ladder_physical[0]), ladder_physical)
    atomic_table(output_root / "ladder_results_alias_expanded.tsv", list(ladder_rows[0]), ladder_rows)
    atomic_table(output_root / "matrix_ladder_rung1_overlap.tsv", list(overlap_rows[0]), overlap_rows)
    atomic_table(output_root / "episode_fingerprints.tsv", list(fingerprint_rows[0]), fingerprint_rows)
    summary = {
        "protocol": PROTOCOL,
        "checkpoint_step": CHECKPOINT_STEP,
        "batch_size": expected_batch_size,
        "batch_count": EPISODE_COUNT // expected_batch_size,
        "episode_count_per_cell": EPISODE_COUNT,
        **counts,
    }
    (output_root / "completeness.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-batch-size", required=True, type=int)
    args = parser.parse_args()
    if args.expected_batch_size <= 0 or EPISODE_COUNT % args.expected_batch_size:
        parser.error("expected-batch-size must divide 512")
    aggregate(
        args.results_root,
        args.output_root,
        expected_batch_size=args.expected_batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
