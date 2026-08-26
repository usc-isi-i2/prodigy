#!/usr/bin/env python3
"""Add completed unconfined-ladder NM evaluations to the schedule comparison."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "data" / "nm_ladder_schedule_comparison_long.csv"
DATASETS = [
    "ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
    "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter",
]


def metric(run_dir: Path) -> float | None:
    paths = sorted((run_dir / "data").glob("metrics_test*.json"), key=lambda p: p.stat().st_mtime)
    for path in reversed(paths):
        try:
            value = json.loads(path.read_text()).get("test_roc_auc")
        except (OSError, json.JSONDecodeError):
            continue
        if value is not None:
            return float(value)
    return None


def find_value(log_root: Path, rung: int, dataset: str) -> float:
    prefix = f"nm_ladder_unconf_h2m_r{rung}"
    pattern = re.compile(rf"^eval_{prefix}_to_{re.escape(dataset)}_nm_3shot_30way")
    runs = sorted(
        (p for p in log_root.glob(f"eval_{prefix}_to_{dataset}_nm_3shot_30way*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run in runs:
        if pattern.match(run.name):
            value = metric(run)
            if value is not None:
                return value
    raise RuntimeError(f"missing completed eval: rung {rung}, {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, required=True)
    args = parser.parse_args()
    with OUTPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 64:
        raise RuntimeError(f"expected 64 paired schedule rows, found {len(rows)}")
    for row in rows:
        row["auc_unconfined"] = find_value(args.log_root, int(row["rung"]), row["test_graph"])
    fields = list(rows[0])
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(OUTPUT)


if __name__ == "__main__":
    main()
