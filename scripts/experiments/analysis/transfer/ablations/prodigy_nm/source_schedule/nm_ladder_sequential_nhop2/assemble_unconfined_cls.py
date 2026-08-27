#!/usr/bin/env python3
"""Assemble the three-arm classification schedule comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
DOWNSTREAM = HERE.parents[1] / "downstream" / "nm_ladder_downstream_nhop2" / "data" / "downstream_long.csv"
OUTPUT = HERE / "data" / "nm_ladder_schedule_cls_comparison_long.csv"
DATASETS = ["covid_political", "election2020", "ukr_rus_suspended", "twibot20"]


def latest_metric(log_root: Path, rung: int, dataset: str) -> tuple[float, str]:
    prefix = f"nm_ladder_unconf_h2m_r{rung}"
    runs = sorted(
        (path for path in log_root.glob(f"eval_{prefix}_to_{dataset}_pl_10shot_*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run in runs:
        metrics = sorted((run / "data").glob("metrics_test*.json"), key=lambda path: path.stat().st_mtime)
        for path in reversed(metrics):
            try:
                value = json.loads(path.read_text()).get("test_roc_auc")
            except (OSError, json.JSONDecodeError):
                continue
            if value is not None:
                return float(value), run.name
    raise RuntimeError(f"missing classification eval: rung {rung}, {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, required=True)
    args = parser.parse_args()
    with DOWNSTREAM.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    lookup = {
        (row["variant"], int(row["rung"]), row["dataset"]): float(row["value"])
        for row in source
        if row["task"] == "classification" and row["metric"] == "roc_auc"
        and row["variant"] in {"matched40k", "sequential"} and row["order"] == "A"
    }
    rows = []
    for rung in range(1, 9):
        for dataset in DATASETS:
            unconfined, eval_run = latest_metric(args.log_root, rung, dataset)
            rows.append({
                "rung": rung,
                "dataset": dataset,
                "auc_interleaved": lookup[("matched40k", rung, dataset)],
                "auc_sequential": lookup[("sequential", rung, dataset)],
                "auc_unconfined": unconfined,
                "eval_run_unconfined": eval_run,
            })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(OUTPUT)


if __name__ == "__main__":
    main()
