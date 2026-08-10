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


def main() -> int:
    expected_manifest, _ = prodigy.validate()
    prodigy.validate_manifest(expected_manifest)
    prodigy_entry = verify_prodigy_entry_effect()
    matrix_cells, ladder_cells, samgpt_entry = verify_samgpt()
    coverage = verify_coverage(matrix_cells, ladder_cells)
    print(
        "FINAL_EXPERIMENT_EVIDENCE_OK "
        f"observed={coverage['observed_total_cells']}/1944 "
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
