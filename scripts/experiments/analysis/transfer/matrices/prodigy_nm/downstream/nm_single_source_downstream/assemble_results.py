#!/usr/bin/env python3
"""Assemble experiment-owned downstream matrices for the eight single-source models."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

CLASS_DATASETS = [
    "covid_political", "election2020", "ukr_rus_suspended", "twibot20",
]
REG_DATASETS = [
    "ukr_rus_twitter", "covid19_twitter", "midterm", "twibot20",
]
REG_TARGETS = ["followers_count", "statuses_count", "account_age_days"]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(here / "data"))
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    manifest_rows = read_csv(data_dir / "model_manifest.csv")
    source_of = {row["model"]: row["source"] for row in manifest_rows}
    source_order = [row["source"] for row in manifest_rows]
    model_of = {row["source"]: row["model"] for row in manifest_rows}
    if len(source_of) != 8:
        raise SystemExit(f"expected 8 models in manifest, found {len(source_of)}")

    class_raw = read_csv(
        data_dir / "parsed/node_classification/data/node_classification.csv"
    )
    reg_raw = read_csv(
        data_dir / "parsed/node_regression/data/node_regression.csv"
    )
    class_raw = [
        row for row in class_raw
        if row["model"] in source_of and row["split"] == "test"
        and row["dataset"] in CLASS_DATASETS and row["shots"] == "10"
    ]
    reg_raw = [
        row for row in reg_raw
        if row["model"] in source_of and row["split"] == "test"
        and row["dataset"] in REG_DATASETS and row["target"] in REG_TARGETS
        and row["shots"] == "10"
    ]

    class_cells = {
        (source_of[row["model"]], row["dataset"]): float(row["roc_auc"])
        for row in class_raw
    }
    reg_cells = {
        (source_of[row["model"]], row["dataset"], row["target"]): float(row["spearman"])
        for row in reg_raw
    }
    missing = []
    for source in source_order:
        for dataset in CLASS_DATASETS:
            if (source, dataset) not in class_cells:
                missing.append(f"classification {source} -> {dataset}")
        for dataset in REG_DATASETS:
            for target in REG_TARGETS:
                if (source, dataset, target) not in reg_cells:
                    missing.append(f"regression {source} -> {dataset}/{target}")
    if missing:
        print(f"ERROR: missing {len(missing)} expected cells")
        for item in missing:
            print(f"  {item}")
        return 1

    class_rows = []
    for source in source_order:
        values = [class_cells[(source, dataset)] for dataset in CLASS_DATASETS]
        class_rows.append({
            "source": source,
            "model": model_of[source],
            **{dataset: class_cells[(source, dataset)] for dataset in CLASS_DATASETS},
            "mean": mean(values),
        })
    write_csv(
        data_dir / "classification.csv",
        ["source", "model", *CLASS_DATASETS, "mean"],
        class_rows,
    )

    reg_rows = []
    reg_dataset_rows = []
    for source in source_order:
        cells = {
            f"{dataset}__{target}": reg_cells[(source, dataset, target)]
            for dataset in REG_DATASETS for target in REG_TARGETS
        }
        reg_rows.append({
            "source": source, "model": model_of[source], **cells,
            "mean": mean(cells.values()),
        })
        dataset_means = {
            dataset: mean(reg_cells[(source, dataset, target)] for target in REG_TARGETS)
            for dataset in REG_DATASETS
        }
        reg_dataset_rows.append({
            "source": source, "model": model_of[source], **dataset_means,
            "mean": mean(dataset_means.values()),
        })
    reg_fields = [
        f"{dataset}__{target}" for dataset in REG_DATASETS for target in REG_TARGETS
    ]
    write_csv(
        data_dir / "regression.csv",
        ["source", "model", *reg_fields, "mean"],
        reg_rows,
    )
    write_csv(
        data_dir / "regression_by_dataset.csv",
        ["source", "model", *REG_DATASETS, "mean"],
        reg_dataset_rows,
    )

    long_rows = []
    for row in class_raw:
        long_rows.append({
            "source": source_of[row["model"]], "model": row["model"],
            "task": "classification", "dataset": row["dataset"], "target": "",
            "metric": "roc_auc", "value": row["roc_auc"], "shots": row["shots"],
            "run": row["run"],
        })
    for row in reg_raw:
        long_rows.append({
            "source": source_of[row["model"]], "model": row["model"],
            "task": "regression", "dataset": row["dataset"], "target": row["target"],
            "metric": "spearman", "value": row["spearman"], "shots": row["shots"],
            "run": row["run"],
        })
    write_csv(
        data_dir / "results_long.csv",
        ["source", "model", "task", "dataset", "target", "metric", "value", "shots", "run"],
        sorted(long_rows, key=lambda row: (
            source_order.index(row["source"]), row["task"], row["dataset"], row["target"]
        )),
    )

    print(f"assembled classification={len(class_cells)} cells")
    print(f"assembled regression={len(reg_cells)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
