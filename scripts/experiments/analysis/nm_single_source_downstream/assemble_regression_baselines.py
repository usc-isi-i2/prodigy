#!/usr/bin/env python3
"""Assemble the two-dataset regression floors into one compact table."""
from __future__ import annotations

import csv
from pathlib import Path


DATASETS = ["ukr_rus_suspended", "twibot20"]
TARGETS = ["followers_count", "statuses_count", "account_age_days"]
BASELINES = ["raw_features", "raw_degree", "random_init"]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    here = Path(__file__).resolve().parent
    data_dir = here / "data"
    cells: dict[tuple[str, str, str], float] = {}

    for baseline, filename in (
        ("raw_features", "regression_baseline_raw_features.csv"),
        ("raw_degree", "regression_baseline_raw_degree.csv"),
    ):
        for row in read_csv(data_dir / filename):
            if row["dataset"] in DATASETS and row["target"] in TARGETS:
                cells[(baseline, row["dataset"], row["target"])] = float(row["spearman"])

    random_path = (
        data_dir / "regression_baseline_random_init_parsed"
        / "node_regression/data/node_regression.csv"
    )
    for row in read_csv(random_path):
        if (
            row["model"] == "random_init"
            and row["dataset"] in DATASETS
            and row["target"] in TARGETS
            and row["split"] == "test"
            and row["shots"] == "10"
        ):
            cells[("random_init", row["dataset"], row["target"])] = float(row["spearman"])

    missing = [
        (baseline, dataset, target)
        for baseline in BASELINES
        for dataset in DATASETS
        for target in TARGETS
        if (baseline, dataset, target) not in cells
    ]
    if missing:
        for item in missing:
            print("missing:", " / ".join(item))
        return 1

    rows = []
    for baseline in BASELINES:
        for dataset in DATASETS:
            values = [cells[(baseline, dataset, target)] for target in TARGETS]
            rows.append({
                "baseline": baseline,
                "dataset": dataset,
                **{target: cells[(baseline, dataset, target)] for target in TARGETS},
                "mean": sum(values) / len(values),
            })

    out = data_dir / "regression_baselines.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["baseline", "dataset", *TARGETS, "mean"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
