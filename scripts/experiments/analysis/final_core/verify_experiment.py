#!/usr/bin/env python3
"""Verify current evidence and coverage for the two-architecture final experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts/experiments/setup/final_core"))

import verify_prodigy_archive as prodigy  # noqa: E402
import build_full_results as full_results  # noqa: E402
from core_plan import ORDERS  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_registered_file(record: dict[str, Any]) -> Path:
    path = REPO / record["path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256(path)
    if observed != record["sha256"]:
        raise ValueError(f"{path}: SHA-256 mismatch")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return list(reader.fieldnames or ()), list(reader)


def verify_samgpt() -> tuple[int, int, int]:
    registry = load_json(HERE / "data/samgpt/observed_seed.json")
    for section in ("matrix", "ladder"):
        verify_registered_file(registry[section]["manifest"])
    matrix_path = verify_registered_file(registry["matrix"]["metrics"])
    ladder_path = verify_registered_file(registry["ladder"]["metrics"])
    verify_registered_file(registry["derived_analysis"]["cells"])
    verify_registered_file(registry["derived_analysis"]["manifest"])

    matrix = read_csv(matrix_path)
    ladder = read_csv(ladder_path)
    matrix_keys = {(row["train_source"], row["target"]) for row in matrix}
    ladder_keys = {(row["order"], int(row["rung"]), row["target"]) for row in ladder}
    if len(matrix) != 81 or len(matrix_keys) != 81:
        raise ValueError("SAMGPT matrix is not an exact 9 x 9 grid")
    if len(ladder) != 243 or len(ladder_keys) != 243:
        raise ValueError("SAMGPT ladder is not an exact 3 x 9 x 9 grid")
    if len({row["train_source"] for row in matrix}) != 9:
        raise ValueError("SAMGPT matrix does not contain nine training sources")
    if len({row["target"] for row in matrix}) != 9:
        raise ValueError("SAMGPT matrix does not contain nine targets")
    if {row["order"] for row in ladder} != {"A", "B", "C"}:
        raise ValueError("SAMGPT ladder does not contain orders A, B, and C")
    for label, rows in (("matrix", matrix), ("ladder", ladder)):
        for row in rows:
            for field in ("loss", "accuracy", "probability_margin"):
                value = float(row[field])
                if not math.isfinite(value):
                    raise ValueError(f"SAMGPT {label}: non-finite {field}")

    loss = {
        (row["order"], int(row["rung"]), row["target"]): float(row["loss"])
        for row in ladder
    }
    improvements = 0
    for order in ("A", "B", "C"):
        added_by_rung = {
            int(row["rung"]): row["added"]
            for row in ladder
            if row["order"] == order
        }
        for rung in range(2, 10):
            target = added_by_rung[rung]
            if loss[(order, rung, target)] < loss[(order, rung - 1, target)]:
                improvements += 1
    if improvements != 21:
        raise ValueError(f"SAMGPT target-entry finding changed: {improvements}/24")
    return len(matrix), len(ladder), improvements


def verify_prodigy_entry_effect() -> int:
    path = HERE / "data/prodigy_final_core/fixed_test/summary/ladder_results_alias_expanded.tsv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    score = {
        (int(row["seed"]), row["order"], int(row["rung"]), row["target"]):
        float(row["score"])
        for row in rows
    }
    improvements = 0
    for seed in (0, 1, 2):
        for order, sources in ORDERS.items():
            for rung, target in enumerate(sources, 1):
                if rung == 1:
                    continue
                if score[(seed, order, rung, target)] > score[(seed, order, rung - 1, target)]:
                    improvements += 1
    if improvements != 72:
        raise ValueError(f"PRODIGY target-entry finding changed: {improvements}/72")
    return improvements


def verify_coverage(matrix_cells: int, ladder_cells: int) -> dict[str, Any]:
    coverage = load_json(HERE / "data/coverage.json")
    design = coverage["design"]
    if design["expected_matrix_cells"] != 486:
        raise ValueError("matrix design is not 2 x 3 x 9 x 9")
    if design["expected_ladder_cells"] != 1458:
        raise ValueError("ladder design is not 2 x 3 x 3 x 9 x 9")
    if coverage["observed_total_cells"] != 243 + 729 + matrix_cells + ladder_cells:
        raise ValueError("coverage total disagrees with the registered evidence")
    if coverage["pending_samgpt_cells"] != 648:
        raise ValueError("pending SAMGPT count is not two matrix-and-ladder seeds")
    return coverage


def verify_full_result_tables() -> tuple[int, int, int]:
    expected = full_results.build_prodigy_rows() + full_results.build_samgpt_rows()
    expected.sort(key=full_results.row_sort_key)
    full_results.validate_rows(expected)

    long_fields, observed_long = read_tsv(HERE / "data/results_full_long.tsv")
    if long_fields != list(full_results.LONG_FIELDS):
        raise ValueError("results_full_long.tsv has the wrong columns or column order")
    if observed_long != expected:
        mismatch = next(
            (
                index
                for index, (observed, wanted) in enumerate(zip(observed_long, expected))
                if observed != wanted
            ),
            min(len(observed_long), len(expected)),
        )
        raise ValueError(f"results_full_long.tsv is stale or incorrect at row {mismatch + 2}")

    wide_fields, observed_wide = read_tsv(HERE / "data/results_full_graphwide.tsv")
    expected_wide = [full_results.wide_row(row) for row in expected]
    if wide_fields != [*full_results.LONG_FIELDS, *full_results.GRAPH_FIELDS]:
        raise ValueError("results_full_graphwide.tsv has the wrong columns or column order")
    if observed_wide != expected_wide:
        mismatch = next(
            (
                index
                for index, (observed, wanted) in enumerate(zip(observed_wide, expected_wide))
                if observed != wanted
            ),
            min(len(observed_wide), len(expected_wide)),
        )
        raise ValueError(
            f"results_full_graphwide.tsv is stale or incorrect at row {mismatch + 2}"
        )

    counts = {
        (architecture, component, status): sum(
            row["architecture"] == architecture
            and row["component"] == component
            and row["result_status"] == status
            for row in expected
        )
        for architecture in ("PRODIGY", "SAMGPT")
        for component in ("matrix", "ladder")
        for status in ("observed", "pending")
    }
    wanted_counts = {
        ("PRODIGY", "matrix", "observed"): 243,
        ("PRODIGY", "matrix", "pending"): 0,
        ("PRODIGY", "ladder", "observed"): 729,
        ("PRODIGY", "ladder", "pending"): 0,
        ("SAMGPT", "matrix", "observed"): 81,
        ("SAMGPT", "matrix", "pending"): 162,
        ("SAMGPT", "ladder", "observed"): 243,
        ("SAMGPT", "ladder", "pending"): 486,
    }
    if counts != wanted_counts:
        raise ValueError(f"full result table coverage is incorrect: {counts}")

    for row in expected:
        if row["result_status"] == "observed":
            source_path = REPO / row["source_result_path"]
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
        if row["aux_result_path"] and not (REPO / row["aux_result_path"]).is_file():
            raise FileNotFoundError(REPO / row["aux_result_path"])

    observed_count = sum(row["result_status"] == "observed" for row in expected)
    return len(expected), observed_count, len(expected) - observed_count


def main() -> int:
    expected_manifest, _ = prodigy.validate()
    prodigy.validate_manifest(expected_manifest)
    prodigy_entry = verify_prodigy_entry_effect()
    matrix_cells, ladder_cells, samgpt_entry = verify_samgpt()
    coverage = verify_coverage(matrix_cells, ladder_cells)
    result_cells, observed_cells, pending_cells = verify_full_result_tables()
    if observed_cells != coverage["observed_total_cells"]:
        raise ValueError("canonical tables disagree with the coverage ledger")
    if pending_cells != coverage["pending_samgpt_cells"]:
        raise ValueError("canonical tables disagree with the pending-cell ledger")
    print(
        "FINAL_EXPERIMENT_EVIDENCE_OK "
        f"observed={observed_cells}/{result_cells} "
        f"prodigy_entry={prodigy_entry}/72 "
        f"samgpt_entry={samgpt_entry}/24 "
        f"samgpt_pending={coverage['pending_samgpt_cells']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
        print(f"FINAL_EXPERIMENT_EVIDENCE_INVALID: {error}", file=sys.stderr)
        raise SystemExit(1)
