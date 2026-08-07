#!/usr/bin/env python3
"""Validate and aggregate the complete 31-model, three-seed final-core grid."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from core_plan import build_models  # noqa: E402


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(results_root: Path, output_root: Path) -> None:
    validation_rows = []
    test_rows = []
    alias_rows = []
    models = build_models()
    for model in models:
        for seed in (0, 1, 2):
            directory = results_root / f"seed_{seed}" / model.model_id
            selection_path = directory / "selection.json"
            result_path = directory / "result.json"
            if not selection_path.is_file() or not result_path.is_file():
                raise FileNotFoundError(f"incomplete result cell: {directory}")
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result["selection_created_utc"] != selection["created_utc"]:
                raise ValueError(f"stale test result: {result_path}")
            if result["selected_checkpoint_step"] != selection["selected_checkpoint_step"]:
                raise ValueError(f"test used a non-selected checkpoint: {result_path}")
            if len(selection["validation_results"]) != 4:
                raise ValueError(f"expected four validation checkpoints: {selection_path}")
            for row in selection["validation_results"]:
                validation_rows.append({
                    "model_id": model.model_id,
                    "seed": seed,
                    "n_sources": len(model.sources),
                    "sources": ",".join(model.sources),
                    "checkpoint_step": row["checkpoint_step"],
                    "validation_score": row["score"],
                    "validation_score_std": row["score_std"],
                    "validation_loss": row["loss"],
                    "selected": int(row["checkpoint_step"] == selection["selected_checkpoint_step"]),
                })
            test = result["test_result"]
            physical = {
                "model_id": model.model_id,
                "seed": seed,
                "n_sources": len(model.sources),
                "sources": ",".join(model.sources),
                "selected_checkpoint_step": result["selected_checkpoint_step"],
                "test_score": test["score"],
                "test_score_std": test["score_std"],
                "test_loss": test["loss"],
            }
            test_rows.append(physical)
            for alias in model.aliases:
                alias_rows.append({"alias": alias, **physical})

    if len(validation_rows) != 31 * 3 * 4 or len(test_rows) != 31 * 3:
        raise AssertionError("unexpected physical grid size")

    summaries = []
    for model in models:
        rows = [row for row in test_rows if row["model_id"] == model.model_id]
        scores = [float(row["test_score"]) for row in rows]
        summaries.append({
            "model_id": model.model_id,
            "n_sources": len(model.sources),
            "sources": ",".join(model.sources),
            "n_seeds": len(rows),
            "test_score_mean": statistics.mean(scores),
            "test_score_sample_std": statistics.stdev(scores),
            "selected_checkpoint_steps": ",".join(str(row["selected_checkpoint_step"]) for row in rows),
        })

    write_tsv(output_root / "checkpoint_validation.tsv", list(validation_rows[0]), validation_rows)
    write_tsv(output_root / "selected_test.tsv", list(test_rows[0]), test_rows)
    write_tsv(output_root / "alias_test.tsv", list(alias_rows[0]), alias_rows)
    write_tsv(output_root / "summary_by_model.tsv", list(summaries[0]), summaries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    aggregate(args.results_root, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
