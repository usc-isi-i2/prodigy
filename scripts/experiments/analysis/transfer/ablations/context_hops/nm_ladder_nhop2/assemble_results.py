#!/usr/bin/env python3
"""Assemble the 2-hop ladder and its paired comparison with the 1-hop ladder.

The eval harness writes one directory per (model, test graph). This script maps the
21 unique 2-hop models back to all 24 (order, rung) rows, emits wide and entry-aligned
long tables, and pairs cells with the committed 1-hop order-robustness table.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())
SETUP = REPO_ROOT / "scripts/experiments/setup/nm_ladder_nhop2"


def load_plan_module():
    path = SETUP / "make_configs.py"
    spec = importlib.util.spec_from_file_location("nm_ladder_nhop2_make_configs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import experiment plan from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLAN = load_plan_module()
DATASETS = list(PLAN.DATASET_KEYS)
KEY_OF_DATASET = {dataset: key for key, dataset, _ in PLAN.SOURCES}


def metric_step(path: Path) -> int:
    match = re.search(r"_step(\d+)\.json$", path.name)
    return int(match.group(1)) if match else -1


def latest_roc_auc(run_dir: Path) -> float | None:
    data_dir = run_dir / "data"
    if not data_dir.is_dir():
        return None
    metrics = sorted(
        data_dir.glob("metrics_test*.json"),
        key=lambda path: (metric_step(path), path.stat().st_mtime),
        reverse=True,
    )
    for path in metrics:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        value = payload.get("test_roc_auc")
        if value is not None:
            return float(value)
    return None


def eval_row(log_root: Path, model_prefix: str) -> tuple[dict[str, float], dict[str, str]]:
    """Return one model's newest AUC and log-dir provenance per test graph."""
    pattern = re.compile(
        rf"^eval_{re.escape(model_prefix)}_to_(?P<test>.+?)_nm_3shot_30way"
    )
    cells: dict[str, float] = {}
    provenance: dict[str, str] = {}
    run_dirs = sorted(
        (path for path in log_root.glob(f"eval_{model_prefix}_to_*_nm_3shot_30way*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    for run_dir in run_dirs:
        match = pattern.match(run_dir.name)
        if match is None or match["test"] not in DATASETS:
            continue
        value = latest_roc_auc(run_dir)
        if value is None:
            continue
        cells[match["test"]] = value
        provenance[match["test"]] = run_dir.name
    return cells, provenance


def read_hop1(path: Path) -> dict[tuple[str, int, str], float]:
    if not path.is_file():
        return {}
    values: dict[tuple[str, int, str], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("auc"):
                continue
            values[(row["order"], int(row["rung"]), row["test_graph"])] = float(row["auc"])
    return values


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def assemble(log_root: Path, orders: set[str] | None = None):
    plan_rows = [
        row for row in PLAN.plan()
        if orders is None or str(row["order"]) in orders
    ]
    prefixes = {str(row["prefix"]) for row in plan_rows}
    model_values: dict[str, dict[str, float]] = {}
    model_provenance: dict[str, dict[str, str]] = {}
    for prefix in sorted(prefixes):
        model_values[prefix], model_provenance[prefix] = eval_row(log_root, prefix)

    wide: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    missing: list[str] = []
    for row in plan_rows:
        order = str(row["order"])
        rung = int(row["rung"])
        prefix = str(row["prefix"])
        cells = model_values[prefix]
        absent = [dataset for dataset in DATASETS if dataset not in cells]
        if absent:
            missing.append(f"{order} r{rung} -> {prefix}: {', '.join(absent)}")

        sources = [str(key) for key in row["sources"]]
        wide.append(
            {
                "n_hop": 2,
                "hop_sizes": "9,9",
                "node_limit": 101,
                "nm_walk_hops": 1,
                "checkpoint_step": 40000,
                "order": order,
                "rung": rung,
                "n_sources": len(sources),
                "added": row["added"],
                "sources": " ".join(sources),
                "model_prefix": prefix,
                "model_status": row["status"],
                **{dataset: cells.get(dataset, "") for dataset in DATASETS},
            }
        )

        sequence = PLAN.ORDERS[order]
        for dataset in DATASETS:
            key = KEY_OF_DATASET[dataset]
            entry_rung = sequence.index(key) + 1
            long_rows.append(
                {
                    "n_hop": 2,
                    "hop_sizes": "9,9",
                    "node_limit": 101,
                    "nm_walk_hops": 1,
                    "checkpoint_step": 40000,
                    "order": order,
                    "rung": rung,
                    "n_sources": len(sources),
                    "test_graph": dataset,
                    "test_canonical": PLAN.canonical(key),
                    "auc": cells.get(dataset, ""),
                    "entry_rung": entry_rung,
                    "rel_to_entry": rung - entry_rung,
                    "in_training": int(rung >= entry_rung),
                    "added": row["added"],
                    "sources": " ".join(sources),
                    "model_prefix": prefix,
                    "model_status": row["status"],
                    "eval_run": model_provenance[prefix].get(dataset, ""),
                }
            )
    return wide, long_rows, missing


def paired_rows(
    hop2_rows: list[dict[str, object]],
    hop1: dict[tuple[str, int, str], float],
) -> list[dict[str, object]]:
    paired: list[dict[str, object]] = []
    for row in hop2_rows:
        if row["auc"] == "":
            continue
        key = (str(row["order"]), int(row["rung"]), str(row["test_graph"]))
        if key not in hop1:
            continue
        hop2_auc = float(row["auc"])
        hop1_auc = hop1[key]
        paired.append(
            {
                "order": row["order"],
                "rung": row["rung"],
                "test_graph": row["test_graph"],
                "entry_rung": row["entry_rung"],
                "rel_to_entry": row["rel_to_entry"],
                "in_training": row["in_training"],
                "auc_h1": hop1_auc,
                "auc_h2": hop2_auc,
                "delta_h2_minus_h1": hop2_auc - hop1_auc,
                "model_prefix_h2m": row["model_prefix"],
            }
        )
    return paired


def print_diagnostics(long_rows: list[dict[str, object]]) -> None:
    complete = [row for row in long_rows if row["auc"] != ""]
    by_key = {
        (str(row["order"]), int(row["rung"]), str(row["test_graph"])): float(row["auc"])
        for row in complete
    }
    jumps = []
    for row in complete:
        if int(row["rel_to_entry"]) != 0 or int(row["entry_rung"]) == 1:
            continue
        before = by_key.get((str(row["order"]), int(row["rung"]) - 1, str(row["test_graph"])))
        if before is not None:
            jumps.append(float(row["auc"]) - before)
    if jumps:
        positive = sum(delta > 0 for delta in jumps)
        mean = sum(jumps) / len(jumps)
        print(f"entry jumps: {positive}/{len(jumps)} positive; mean={mean:+.4f}")
    else:
        print("entry jumps: unavailable (results incomplete)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=Path("/dataMeR1/phil/gfm/prodigy-nmlh2/log"))
    parser.add_argument("--out-dir", type=Path, default=HERE / "data")
    parser.add_argument(
        "--phase", choices=["A", "all"], default="all",
        help="assemble canonical Order A only, or require all three orders",
    )
    parser.add_argument(
        "--hop1-long", type=Path,
        default=REPO_ROOT / "scripts/experiments/analysis/transfer/ladders/prodigy_nm/order_and_graph_set/nm_ladder_order_robustness/data/nm_ladder_order_robustness_long.csv",
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    orders = {"A"} if args.phase == "A" else None
    suffix = "_order_A" if args.phase == "A" else ""
    wide, long_rows, missing = assemble(args.log_root, orders=orders)
    write_csv(
        args.out_dir / f"nm_ladder_nhop2{suffix}.csv",
        [
            "n_hop", "hop_sizes", "node_limit", "nm_walk_hops", "checkpoint_step",
            "order", "rung", "n_sources", "added", "sources", "model_prefix",
            "model_status", *DATASETS,
        ],
        wide,
    )
    long_fields = [
        "n_hop", "hop_sizes", "node_limit", "nm_walk_hops", "checkpoint_step",
        "order", "rung", "n_sources", "test_graph", "test_canonical", "auc",
        "entry_rung", "rel_to_entry", "in_training", "added", "sources",
        "model_prefix", "model_status", "eval_run",
    ]
    write_csv(args.out_dir / f"nm_ladder_nhop2{suffix}_long.csv", long_fields, long_rows)

    hop1 = read_hop1(args.hop1_long)
    if hop1:
        comparison = paired_rows(long_rows, hop1)
        write_csv(
            args.out_dir / f"nm_ladder_nhop_comparison{suffix}_long.csv",
            [
                "order", "rung", "test_graph", "entry_rung", "rel_to_entry",
                "in_training", "auc_h1", "auc_h2", "delta_h2_minus_h1",
                "model_prefix_h2m",
            ],
            comparison,
        )
    else:
        print(f"WARN: no 1-hop long table at {args.hop1_long}; comparison not written")

    print_diagnostics(long_rows)
    if missing:
        print(f"{len(missing)} order/rung rows have missing test cells", file=sys.stderr)
        for item in missing[:24]:
            print(f"  {item}", file=sys.stderr)
        if not args.allow_partial:
            print("exiting 1; pass --allow-partial to accept incomplete results", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
