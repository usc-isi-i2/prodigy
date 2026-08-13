#!/usr/bin/env python3
"""Assemble classification and repaired static-LP results for all h2 ladders."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())
SETUP = REPO_ROOT / "scripts/experiments/setup/nm_ladder_downstream_nhop2"

CLASSIFICATION_DATASETS = (
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
)
STATIC_LP_DATASETS = (
    "ukr_rus_twitter",
    "covid19_twitter",
    "midterm",
    "twibot20",
    "cp_hk_twitter",
)
KEY_OF_DATASET = {
    "ukr_rus_twitter": "ukr_rus",
    "covid19_twitter": "covid",
    "midterm": "midterm",
    "covid_political": "covid_political",
    "election2020": "election2020",
    "ukr_rus_suspended": "ukr_rus_suspended",
    "twibot20": "twibot20",
    "cp_hk_twitter": "cp_hk",
}
PRIMARY = {"classification": "roc_auc", "static_lp": "auc"}
EXPECTED_FLOORS = {
    "common_neighbors",
    "adamic_adar",
    "preferential_attachment",
    "jaccard",
    "raw_feature_cosine",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_classification(path: Path, models: set[str]) -> tuple[dict[tuple[str, str], dict], list[str]]:
    cells: dict[tuple[str, str], dict] = {}
    duplicates: list[str] = []
    for row in read_csv(path):
        if row.get("model") not in models:
            continue
        if row.get("split") != "test" or str(row.get("shots")) != "10":
            continue
        dataset = row.get("dataset", "")
        if dataset not in CLASSIFICATION_DATASETS:
            continue
        key = (row["model"], dataset)
        if key in cells:
            duplicates.append(f"classification {key[0]}@{key[1]}")
            continue
        cells[key] = {
            metric: to_float(row.get(metric)) for metric in ("roc_auc", "accuracy", "f1")
        }
    return cells, duplicates


def load_pair_lp(pair_dir: Path, models: set[str]) -> tuple[dict[tuple[str, str], dict], list[dict], list[str]]:
    cells: dict[tuple[str, str], dict] = {}
    floors: list[dict] = []
    errors: list[str] = []
    for dataset in STATIC_LP_DATASETS:
        path = pair_dir / f"{dataset}__pair_lp.csv"
        if not path.is_file():
            errors.append(f"missing {path}")
            continue
        dataset_floors: list[str] = []
        for row in read_csv(path):
            if row.get("negative_kind") != "degree_matched":
                continue
            if row.get("model") == "__floor__":
                dataset_floors.append(row.get("scorer", ""))
                floors.append({
                    "dataset": dataset,
                    "scorer": row.get("scorer", ""),
                    "auc": row.get("auc", ""),
                    "average_precision": row.get("average_precision", ""),
                    "hits_at_50": row.get("hits_at_50", ""),
                })
                continue
            model = row.get("model", "")
            if model not in models or row.get("scorer") != "encoder_cosine":
                continue
            key = (model, dataset)
            if key in cells:
                errors.append(f"duplicate static_lp {model}@{dataset}")
                continue
            leak = to_float(row.get("leakage_edges"))
            sensitivity = to_float(row.get("endpoint_sensitivity"))
            permutation = to_float(row.get("endpoint_permutation_auc"))
            if (
                leak != 0
                or sensitivity is None
                or sensitivity < 0.99
                or permutation is None
                or abs(permutation - 0.5) >= 0.05
            ):
                errors.append(
                    f"invalid static_lp {model}@{dataset}: leak={leak}, "
                    f"sensitivity={sensitivity}, permutation={permutation}"
                )
                continue
            cells[key] = {
                metric: to_float(row.get(metric))
                for metric in ("auc", "average_precision", "hits_at_50")
            }
        missing_floors = sorted(EXPECTED_FLOORS - set(dataset_floors))
        duplicate_floors = sorted(
            scorer for scorer in set(dataset_floors) if dataset_floors.count(scorer) != 1
        )
        if missing_floors:
            errors.append(f"{dataset}: missing floors: {', '.join(missing_floors)}")
        if duplicate_floors:
            errors.append(f"{dataset}: duplicate floors: {', '.join(duplicate_floors)}")
    return cells, floors, errors


def _entry_rungs(row_map: list[dict[str, str]]) -> dict[tuple[str, str, str], int]:
    entries: dict[tuple[str, str, str], int] = {}
    for row in row_map:
        sources = set(row["sources"].split(","))
        for key in KEY_OF_DATASET.values():
            if key in sources:
                index = (row["variant"], row["order"], key)
                entries[index] = min(entries.get(index, 10**9), int(row["rung"]))
    return entries


def build_long_rows(
    row_map: list[dict[str, str]],
    classification: dict[tuple[str, str], dict],
    static_lp: dict[tuple[str, str], dict],
) -> tuple[list[dict], list[str]]:
    entries = _entry_rungs(row_map)
    long_rows: list[dict] = []
    missing: list[str] = []
    tasks = (
        ("classification", CLASSIFICATION_DATASETS, classification),
        ("static_lp", STATIC_LP_DATASETS, static_lp),
    )
    for logical in row_map:
        rung = int(logical["rung"])
        sources = set(logical["sources"].split(","))
        model = logical["model_key"]
        for task, datasets, cells in tasks:
            for dataset in datasets:
                values = cells.get((model, dataset))
                if values is None:
                    missing.append(f"{task} {model}@{dataset}")
                    continue
                graph_key = KEY_OF_DATASET[dataset]
                entry = entries[(logical["variant"], logical["order"], graph_key)]
                base = {
                    **logical,
                    "task": task,
                    "dataset": dataset,
                    "entry_rung": entry,
                    "rel_to_entry": rung - entry,
                    "in_training": int(graph_key in sources),
                    "is_newcomer": int(rung == entry),
                }
                for metric, value in values.items():
                    if value is not None:
                        long_rows.append({
                            **base,
                            "metric": metric,
                            "value": value,
                            "primary": int(metric == PRIMARY[task]),
                        })
    return long_rows, missing


def entry_jumps(long_rows: list[dict]) -> list[dict]:
    primary = {
        (row["variant"], row["order"], row["task"], row["dataset"], int(row["rung"])): row
        for row in long_rows
        if int(row["primary"])
    }
    jumps: list[dict] = []
    for row in primary.values():
        entry = int(row["entry_rung"])
        if int(row["rung"]) != entry or entry == 1:
            continue
        before_key = (row["variant"], row["order"], row["task"], row["dataset"], entry - 1)
        before = primary.get(before_key)
        if before is None:
            continue
        jumps.append({
            "variant": row["variant"],
            "order": row["order"],
            "task": row["task"],
            "dataset": row["dataset"],
            "entry_rung": entry,
            "before": before["value"],
            "after": row["value"],
            "delta": float(row["value"]) - float(before["value"]),
            "before_model": before["model_key"],
            "after_model": row["model_key"],
        })
    return sorted(jumps, key=lambda row: (
        row["task"], row["variant"], row["order"], row["entry_rung"], row["dataset"]
    ))


def pair_to_control(long_rows: list[dict]) -> list[dict]:
    primary = [row for row in long_rows if int(row["primary"])]
    baseline = {
        (int(row["rung"]), row["task"], row["dataset"]): row
        for row in primary
        if row["variant"] == "matched40k" and row["order"] == "A"
    }
    paired: list[dict] = []
    for row in primary:
        if row["variant"] not in {"sequential", "split", "fixed10k"} or row["order"] != "A":
            continue
        key = (int(row["rung"]), row["task"], row["dataset"])
        control = baseline.get(key)
        if control is None:
            continue
        role = "newcomer" if int(row["is_newcomer"]) else (
            "incumbent" if int(row["in_training"]) else "heldout"
        )
        paired.append({
            "variant": row["variant"],
            "order": row["order"],
            "rung": row["rung"],
            "task": row["task"],
            "dataset": row["dataset"],
            "role": role,
            "control_value": control["value"],
            "variant_value": row["value"],
            "delta_vs_matched40k": float(row["value"]) - float(control["value"]),
            "control_model": control["model_key"],
            "variant_model": row["model_key"],
        })
    return sorted(paired, key=lambda row: (
        row["task"], row["variant"], int(row["rung"]), row["dataset"]
    ))


def summarize(jumps: list[dict], paired: list[dict]) -> dict:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in jumps:
        grouped[(row["variant"], row["order"], row["task"])].append(float(row["delta"]))
    entry = []
    for (variant, order, task), values in sorted(grouped.items()):
        entry.append({
            "variant": variant,
            "order": order,
            "task": task,
            "n": len(values),
            "positive": sum(value > 0 for value in values),
            "mean_delta": sum(values) / len(values),
            "min_delta": min(values),
            "max_delta": max(values),
        })
    return {
        "logical_rows": 40,
        "physical_models": 39,
        "entry_jumps": entry,
        "paired_cells_to_matched40k": len(paired),
        "notes": [
            "One training seed; cell counts are paired measurements, not independent replicates.",
            "Static LP primary negative set is degree_matched and uses the repaired pair evaluator.",
            "Temporal LP is excluded because its evaluator remains invalid.",
        ],
    }


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows and fields is None:
        raise ValueError(f"cannot infer fields for empty output {path}")
    fieldnames = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def wide_rows(long_rows: list[dict], task: str, datasets: tuple[str, ...]) -> list[dict]:
    selected = [row for row in long_rows if row["task"] == task and int(row["primary"])]
    by_logical: dict[str, dict] = {}
    for row in selected:
        logical = by_logical.setdefault(row["logical_id"], {
            key: row[key] for key in (
                "logical_id", "variant", "order", "rung", "added", "sources",
                "model_key", "checkpoint_step", "schedule", "exposure", "train_edges"
            )
        })
        logical[row["dataset"]] = row["value"]
    return [
        {**row, **{dataset: row.get(dataset, "") for dataset in datasets}}
        for row in sorted(by_logical.values(), key=lambda item: (
            item["variant"], item["order"], int(item["rung"])
        ))
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-map", type=Path, default=SETUP / "row_map.csv")
    parser.add_argument(
        "--classification-csv",
        type=Path,
        default=HERE / "data/raw/runner/node_classification/data/node_classification.csv",
    )
    parser.add_argument("--pair-dir", type=Path, default=HERE / "data/raw/pair_lp")
    parser.add_argument("--out-dir", type=Path, default=HERE / "data")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    for path in (args.row_map, args.classification_csv):
        if not path.is_file():
            print(f"ERROR: missing input {path}", file=sys.stderr)
            return 2
    rows = read_csv(args.row_map)
    models = {row["model_key"] for row in rows}
    if len(rows) != 40 or len(models) != 39:
        print(
            f"ERROR: row map must contain 40 logical rows / 39 models; got "
            f"{len(rows)} / {len(models)}",
            file=sys.stderr,
        )
        return 2

    classification, classification_errors = load_classification(args.classification_csv, models)
    static_lp, floors, pair_errors = load_pair_lp(args.pair_dir, models)
    long_rows, missing = build_long_rows(rows, classification, static_lp)
    errors = classification_errors + pair_errors + missing
    if errors and not args.allow_partial:
        print(f"ERROR: {len(errors)} result error(s); no outputs written", file=sys.stderr)
        for error in errors[:80]:
            print(f"  {error}", file=sys.stderr)
        return 1

    jumps = entry_jumps(long_rows)
    paired = pair_to_control(long_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "downstream_long.csv", long_rows)
    write_csv(args.out_dir / "entry_jumps.csv", jumps)
    write_csv(args.out_dir / "paired_to_matched40k.csv", paired)
    write_csv(
        args.out_dir / "pair_lp_floors.csv",
        floors,
        ["dataset", "scorer", "auc", "average_precision", "hits_at_50"],
    )
    write_csv(
        args.out_dir / "classification_roc_auc.csv",
        wide_rows(long_rows, "classification", CLASSIFICATION_DATASETS),
    )
    write_csv(
        args.out_dir / "static_lp_auc.csv",
        wide_rows(long_rows, "static_lp", STATIC_LP_DATASETS),
    )
    summary = summarize(jumps, paired)
    summary["errors_allowed"] = errors
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"assembled {len(long_rows)} metric rows; {len(jumps)} entry jumps; "
        f"{len(paired)} controlled pairs"
    )
    if errors:
        print(f"WARNING: partial output with {len(errors)} error(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
