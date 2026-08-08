#!/usr/bin/env python3
"""Strict aggregation for the completed three-arm, three-seed radius experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from radius_plan import ARMS, CHECKPOINT_STEPS, PANELS  # noqa: E402


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"refusing empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validation_rows = []
    test_rows = []
    selections = []
    for arm in ARMS:
        for seed in (0, 1, 2):
            directory = args.results_root / f"seed_{seed}" / arm.arm_id
            selection_path = directory / "selection.json"
            result_path = directory / "result.json"
            if not selection_path.is_file() or not result_path.is_file():
                raise FileNotFoundError(f"missing frozen result pair under {directory}")
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result["selection_created_utc"] != selection["created_utc"]:
                raise ValueError(f"selection/result mismatch under {directory}")
            selections.append(
                {
                    "arm": arm.arm_id,
                    "seed": seed,
                    "selected_checkpoint_step": selection["selected_checkpoint_step"],
                }
            )
            for row in selection["validation_results"]:
                validation_rows.append(
                    {
                        "arm": arm.arm_id,
                        "seed": seed,
                        "checkpoint_step": row["checkpoint_step"],
                        "panel": row["panel"],
                        "score": row["score"],
                        "score_std": row["score_std"],
                        "loss": row["loss"],
                    }
                )
            for row in result["test_results"]:
                test_rows.append(
                    {
                        "arm": arm.arm_id,
                        "seed": seed,
                        "selected_checkpoint_step": result["selected_checkpoint_step"],
                        "panel": row["panel"],
                        "score": row["score"],
                        "score_std": row["score_std"],
                        "loss": row["loss"],
                    }
                )

    expected_validation = len(ARMS) * 3 * len(CHECKPOINT_STEPS) * 3
    expected_test = len(ARMS) * 3 * len(PANELS)
    if len(validation_rows) != expected_validation or len(test_rows) != expected_test:
        raise ValueError(
            f"unexpected row counts: validation={len(validation_rows)}/{expected_validation}, "
            f"test={len(test_rows)}/{expected_test}"
        )

    summary = []
    for arm in ARMS:
        for panel in PANELS:
            scores = [
                float(row["score"])
                for row in test_rows
                if row["arm"] == arm.arm_id and row["panel"] == panel.panel_id
            ]
            summary.append(
                {
                    "arm": arm.arm_id,
                    "panel": panel.panel_id,
                    "mean_score": statistics.mean(scores),
                    "seed_std": statistics.stdev(scores),
                    "scores": scores,
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "validation_trajectory.csv", validation_rows)
    write_csv(args.output_root / "test_results.csv", test_rows)
    write_csv(args.output_root / "checkpoint_selections.csv", selections)
    (args.output_root / "summary.json").write_text(
        json.dumps(
            {
                "protocol": "nm_all9_radius_finalcore_summary_v1",
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
