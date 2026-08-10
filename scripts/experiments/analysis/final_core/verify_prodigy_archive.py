#!/usr/bin/env python3
"""Validate the committed final-core evidence package and its checksums."""

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
DATA = HERE / "data/prodigy_final_core"
SOURCES = (
    "ukr_rus",
    "covid",
    "midterm",
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
    "cp_hk",
    "facebook_page_reference",
)
SEEDS = (0, 1, 2)
PROTOCOL = "fixed_test_512_static_test_on_static_train_v1"
METRIC_CONTRACT = "accuracy_f1_macro_roc_auc_ovr_macro_v1"


def fail(message: str) -> None:
    raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"{path}: invalid JSON: {error}")
    if not isinstance(value, dict):
        fail(f"{path}: top-level JSON value is not an object")
    return value


def assert_finite(value: Any, path: Path, field: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        fail(f"{path}: non-finite value at {field}")
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite(child, path, f"{field}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite(child, path, f"{field}[{index}]")


def result_key(path: Path, root: Path) -> tuple[int, str, str]:
    relative = path.relative_to(root)
    if len(relative.parts) != 3:
        fail(f"{path}: unexpected result path")
    seed_part, model_id, filename = relative.parts
    if not seed_part.startswith("seed_") or not filename.endswith(".json"):
        fail(f"{path}: unexpected result path")
    return int(seed_part.removeprefix("seed_")), model_id, filename.removesuffix(".json")


def load_results(root: Path, expected_count: int) -> dict[tuple[int, str, str], dict[str, Any]]:
    paths = sorted(root.glob("seed_*/*/*.json"))
    if len(paths) != expected_count:
        fail(f"{root}: expected {expected_count} JSON files, found {len(paths)}")
    results: dict[tuple[int, str, str], dict[str, Any]] = {}
    for path in paths:
        key = result_key(path, root)
        if key in results:
            fail(f"{path}: duplicate result key {key}")
        payload = load_json(path)
        assert_finite(payload, path)
        seed, model_id, target = key
        checks = {
            "seed": seed,
            "model_id": model_id,
            "target": target,
            "protocol": PROTOCOL,
            "checkpoint_step": 2500,
            "episode_count": 512,
            "batch_size": 32,
            "batch_count": 16,
            "split": "test",
            "edge_view": "static_train",
            "target_edge_view": "static_test",
        }
        for field, expected in checks.items():
            if payload.get(field) != expected:
                fail(f"{path}: {field}={payload.get(field)!r}, expected {expected!r}")
        for field in ("episode_plan_fingerprint", "observed_episode_fingerprint"):
            value = payload.get(field)
            if not isinstance(value, str) or len(value) != 64:
                fail(f"{path}: invalid {field}")
        results[key] = payload
    return results


def load_fingerprint_ledger() -> dict[str, tuple[str, str]]:
    path = DATA / "auc/reference/episode_fingerprints.tsv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if {row["target"] for row in rows} != set(SOURCES) or len(rows) != len(SOURCES):
        fail(f"{path}: expected exactly the nine target rows")
    return {
        row["target"]: (
            row["episode_plan_fingerprint"],
            row["observed_episode_fingerprint"],
        )
        for row in rows
    }


def validate_fingerprints(
    label: str,
    results: dict[tuple[int, str, str], dict[str, Any]],
    ledger: dict[str, tuple[str, str]],
) -> None:
    for (_, _, target), payload in results.items():
        observed = (
            payload["episode_plan_fingerprint"],
            payload["observed_episode_fingerprint"],
        )
        if observed != ledger[target]:
            fail(f"{label}: fingerprint mismatch for target {target}")


def validate_auc(
    auc: dict[tuple[int, str, str], dict[str, Any]],
) -> None:
    expected = {
        (seed, f"ss_{source}", target)
        for seed in SEEDS
        for source in SOURCES
        for target in SOURCES
    }
    if set(auc) != expected:
        fail("AUC specialist grid is not the exact 3 x 9 x 9 contract")
    for key, payload in auc.items():
        if payload.get("metric_contract") != METRIC_CONTRACT:
            fail(f"AUC cell {key}: wrong metric contract")
        for field in ("accuracy", "f1_macro", "roc_auc_ovr_macro"):
            value = payload.get(field)
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                fail(f"AUC cell {key}: invalid {field}={value!r}")
        if payload["score"] != payload["accuracy"]:
            fail(f"AUC cell {key}: score and accuracy disagree")


def validate_auc_long_table(
    auc: dict[tuple[int, str, str], dict[str, Any]],
) -> None:
    path = DATA / "auc/summary/single_source_metrics_long.tsv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 243:
        fail(f"{path}: expected 243 data rows, found {len(rows)}")
    seen = set()
    for row in rows:
        key = (int(row["seed"]), f"ss_{row['source']}", row["target"])
        if key in seen or key not in auc:
            fail(f"{path}: duplicate or unexpected key {key}")
        seen.add(key)
        payload = auc[key]
        for field in ("accuracy", "f1_macro", "roc_auc_ovr_macro"):
            if not math.isclose(float(row[field]), float(payload[field]), rel_tol=0, abs_tol=1e-15):
                fail(f"{path}: {field} disagrees with raw cell {key}")
    if seen != set(auc):
        fail(f"{path}: long table does not cover every AUC cell")


def read_matrix(path: Path) -> dict[tuple[str, str], float]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 9 or set(rows[0] if rows else ()) != {"model_source", *SOURCES}:
        fail(f"{path}: expected a labelled 9 x 9 matrix")
    if {row["model_source"] for row in rows} != set(SOURCES):
        fail(f"{path}: source labels do not match the graph registry")
    return {
        (row["model_source"], target): float(row[target])
        for row in rows
        for target in SOURCES
    }


def validate_auc_matrices(
    auc: dict[tuple[int, str, str], dict[str, Any]],
) -> None:
    root = DATA / "auc/summary"
    for metric in ("accuracy", "f1_macro", "roc_auc_ovr_macro"):
        seed_values: dict[int, dict[tuple[str, str], float]] = {}
        for seed in SEEDS:
            path = root / f"single_source_{metric}_seed_{seed}.csv"
            observed = read_matrix(path)
            expected = {
                (source, target): float(auc[(seed, f"ss_{source}", target)][metric])
                for source in SOURCES
                for target in SOURCES
            }
            for key in expected:
                if not math.isclose(observed[key], expected[key], rel_tol=0, abs_tol=1e-15):
                    fail(f"{path}: value {key} disagrees with its raw result")
            seed_values[seed] = expected
        expected_mean = {
            key: statistics.mean(seed_values[seed][key] for seed in SEEDS)
            for key in seed_values[0]
        }
        expected_std = {
            key: statistics.stdev(seed_values[seed][key] for seed in SEEDS)
            for key in seed_values[0]
        }
        for suffix, expected in (
            ("three_seed_mean", expected_mean),
            ("three_seed_sample_std", expected_std),
        ):
            path = root / f"single_source_{metric}_{suffix}.csv"
            observed = read_matrix(path)
            for key in expected:
                if not math.isclose(observed[key], expected[key], rel_tol=0, abs_tol=1e-15):
                    fail(f"{path}: aggregate {key} disagrees with the seed matrices")


def validate_fixed_topology(
    fixed: dict[tuple[int, str, str], dict[str, Any]],
) -> None:
    plan_path = DATA / "fixed_test/physical_plan.tsv"
    with plan_path.open(encoding="utf-8", newline="") as handle:
        jobs = list(csv.DictReader(handle, delimiter="\t"))
    if len(jobs) != 93:
        fail(f"{plan_path}: expected 93 checkpoint-seed jobs, found {len(jobs)}")
    expected = {
        (int(row["seed"]), row["model_id"], target)
        for row in jobs
        for target in SOURCES
    }
    if set(fixed) != expected:
        fail("fixed-test grid does not equal physical_plan.tsv x nine targets")

    summary = DATA / "fixed_test/summary"
    expected_rows = {
        "combined_physical_cells.tsv": 837,
        "single_source_matrix_long.tsv": 243,
        "ladder_physical_cells.tsv": 675,
        "ladder_results_alias_expanded.tsv": 729,
        "matrix_ladder_rung1_overlap.tsv": 81,
        "episode_fingerprints.tsv": 9,
    }
    for filename, expected_count in expected_rows.items():
        path = summary / filename
        with path.open(encoding="utf-8", newline="") as handle:
            row_count = sum(1 for _ in csv.DictReader(handle, delimiter="\t"))
        if row_count != expected_count:
            fail(f"{path}: expected {expected_count} rows, found {row_count}")

    for seed in SEEDS:
        path = summary / f"single_source_matrix_seed_{seed}.csv"
        observed = read_matrix(path)
        for source in SOURCES:
            for target in SOURCES:
                expected_score = float(fixed[(seed, f"ss_{source}", target)]["score"])
                if not math.isclose(
                    observed[(source, target)], expected_score, rel_tol=0, abs_tol=1e-15
                ):
                    fail(f"{path}: value {(source, target)} disagrees with raw result")


def validate_cross_run_replay(
    fixed: dict[tuple[int, str, str], dict[str, Any]],
    auc: dict[tuple[int, str, str], dict[str, Any]],
) -> dict[str, Any]:
    deltas = []
    for key, auc_payload in auc.items():
        fixed_payload = fixed.get(key)
        if fixed_payload is None:
            fail(f"fixed-test archive is missing specialist cell {key}")
        deltas.append(abs(float(auc_payload["accuracy"]) - float(fixed_payload["score"])))
    max_delta = max(deltas)
    one_prediction = 1.0 / 61_440.0
    if max_delta > one_prediction + 1e-15:
        fail(f"cross-run specialist accuracy delta {max_delta} exceeds one prediction")
    return {
        "cells_compared": len(deltas),
        "bit_identical_cells": sum(delta == 0 for delta in deltas),
        "one_prediction_delta_cells": sum(delta != 0 for delta in deltas),
        "max_absolute_accuracy_delta": max_delta,
    }


def data_files() -> list[Path]:
    manifest = DATA / "manifest.json"
    return sorted(path for path in DATA.rglob("*") if path.is_file() and path != manifest)


def file_record(path: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "path": path.relative_to(DATA).as_posix(),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def archive_digest(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            f"{record['path']}\0{record['bytes']}\0{record['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def build_manifest(reconciliation: dict[str, Any]) -> dict[str, Any]:
    records = [file_record(path) for path in data_files()]
    return {
        "archive_contract": "prodigy_final_core_evidence_archive_v1",
        "architecture": "PRODIGY",
        "protocol": PROTOCOL,
        "producing_commits": {
            "fixed_test": "c5be3b9022d0f8638525e138050c11472fe05d60",
            "auc_supplement": "62d9e3141ab4b1fa2f6002ae85d03e32bba7f063",
        },
        "source_locations": {
            "training_state": "/dataMeR1/phil/gfm/prodigy-final-core/state/final_core",
            "fixed_test": "/dataMeR1/phil/gfm/prodigy-final-core-fixed-test/log/final_core_fixed_test/production/bs32",
            "auc_supplement": "/dataMeR1/phil/gfm/prodigy-final-core-auc/log/final_core_auc/production/bs32",
            "fingerprint_ledger": "/dataMeR1/phil/gfm/prodigy-final-core-cache/log/final_core_cached_test/production/bs32/summary/episode_fingerprints.tsv",
        },
        "counts": {
            "fixed_test_physical_cells": 837,
            "fixed_test_alias_expanded_ladder_rows": 729,
            "auc_specialist_cells": 243,
            "sources": 9,
            "targets": 9,
            "training_seeds": 3,
        },
        "cross_run_reconciliation": reconciliation,
        "file_count": len(records),
        "archive_sha256": archive_digest(records),
        "files": records,
    }


def validate_manifest(expected: dict[str, Any]) -> None:
    path = DATA / "manifest.json"
    actual = load_json(path)
    if actual != expected:
        fail(f"{path}: content or checksums are stale; regenerate with --write-manifest")


def validate() -> tuple[dict[str, Any], dict[str, Any]]:
    fixed = load_results(DATA / "fixed_test/results", 837)
    auc = load_results(DATA / "auc/results", 243)
    validate_fixed_topology(fixed)
    validate_auc(auc)
    validate_auc_long_table(auc)
    validate_auc_matrices(auc)
    ledger = load_fingerprint_ledger()
    validate_fingerprints("fixed test", fixed, ledger)
    validate_fingerprints("AUC supplement", auc, ledger)
    reconciliation = validate_cross_run_replay(fixed, auc)
    return build_manifest(reconciliation), reconciliation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="replace data/manifest.json after all semantic checks pass",
    )
    args = parser.parse_args()
    expected, reconciliation = validate()
    manifest_path = DATA / "manifest.json"
    if args.write_manifest:
        manifest_path.write_text(
            json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        validate_manifest(expected)
    print(
        "PRODIGY_FINAL_CORE_ARCHIVE_OK "
        f"fixed_cells=837 auc_cells=243 files={expected['file_count']} "
        f"replay_exact={reconciliation['bit_identical_cells']}/243 "
        f"sha256={expected['archive_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"PRODIGY_FINAL_CORE_ARCHIVE_INVALID: {error}", file=sys.stderr)
        raise SystemExit(1)
