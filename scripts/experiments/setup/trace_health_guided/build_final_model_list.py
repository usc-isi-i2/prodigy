#!/usr/bin/env python3
"""Combine r4 schedule, specialist/merge, and health-guided checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scripts.experiments.setup.trace_health_guided.make_stage1_plan import SOURCES


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-model-list", type=Path, required=True)
    parser.add_argument("--stage1-model-list", type=Path, required=True)
    parser.add_argument("--health-model-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite model list: {args.output}")
    schedule = [
        row
        for row in read_rows(args.schedule_model_list)
        if row["model_id"].startswith("r4_")
    ]
    stage1 = read_rows(args.stage1_model_list)
    health = read_rows(args.health_model_list)
    rows = schedule + stage1 + health
    expected_schedule = {
        f"r4_{condition}_s{seed}"
        for condition in ("blocked", "replay100", "interleaved")
        for seed in range(3)
    }
    expected_stage1 = {
        *(f"ss_{source}_s{seed}" for source in SOURCES for seed in range(3)),
        *(f"merged_s{seed}" for seed in range(3)),
    }
    expected_health = {f"health_guided_s{seed}" for seed in range(3)}
    actual = {row["model_id"] for row in rows}
    expected = expected_schedule | expected_stage1 | expected_health
    if actual != expected or len(rows) != len(expected):
        raise ValueError(
            f"final model lattice mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}, rows={len(rows)}"
        )
    missing = [row["checkpoint"] for row in rows if not Path(row["checkpoint"]).is_file()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["model_id", "checkpoint", "sources"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"OK: wrote {len(rows)} final comparison checkpoints to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
