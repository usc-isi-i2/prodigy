#!/usr/bin/env python3
"""Assemble the complete split-aware eight-rung NM ladder from eval logs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SETUP = HERE.parents[1] / "setup" / "nm_ladder_train_test_nhop2"
DATASETS = [
    "ukr_rus_twitter", "covid19_twitter", "midterm", "covid_political",
    "election2020", "ukr_rus_suspended", "twibot20", "cp_hk_twitter",
]
DATASET_TO_SOURCE = {
    "ukr_rus_twitter": "ukr_rus",
    "covid19_twitter": "covid",
    "midterm": "midterm",
    "covid_political": "covid_political",
    "election2020": "election2020",
    "ukr_rus_suspended": "ukr_rus_suspended",
    "twibot20": "twibot20",
    "cp_hk_twitter": "cp_hk",
}


def load_plan():
    path = SETUP / "make_configs.py"
    spec = importlib.util.spec_from_file_location("nm_ladder_tts_plan", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLAN = load_plan()


def metric_step(path: Path) -> int:
    match = re.search(r"_step(\d+)\.json$", path.name)
    return int(match.group(1)) if match else -1


def latest_auc(run_dir: Path) -> float | None:
    data = run_dir / "data"
    if not data.is_dir():
        return None
    paths = sorted(
        data.glob("metrics_test*.json"),
        key=lambda path: (metric_step(path), path.stat().st_mtime),
        reverse=True,
    )
    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("test_roc_auc") is not None:
            return float(payload["test_roc_auc"])
    return None


def eval_row(log_root: Path, prefix: str):
    pattern = re.compile(
        rf"^eval_{re.escape(prefix)}_to_(?P<test>.+?)_nm_3shot_30way"
    )
    values, provenance = {}, {}
    runs = sorted(
        (p for p in log_root.glob(f"eval_{prefix}_to_*_nm_3shot_30way*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    for run in runs:
        match = pattern.match(run.name)
        if match is None or match["test"] not in DATASETS:
            continue
        auc = latest_auc(run)
        if auc is not None:
            values[match["test"]] = auc
            provenance[match["test"]] = run.name
    return values, provenance


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def assemble(log_root: Path):
    wide, long_rows, missing = [], [], []
    for row in PLAN.plan():
        prefix = str(row["prefix"])
        cells, provenance = eval_row(log_root, prefix)
        absent = [dataset for dataset in DATASETS if dataset not in cells]
        if absent:
            missing.append(f"r{row['rung']} {prefix}: {', '.join(absent)}")
        sources = [str(source) for source in row["sources"]]
        common = {
            "rung": int(row["rung"]),
            "n_sources": len(sources),
            "added": row["added"],
            "sources": " ".join(sources),
            "model_prefix": prefix,
            "checkpoint_step": 40000,
            "n_hop": 2,
            "hop_sizes": "9,9",
            "node_limit": 101,
            "nm_walk_hops": 1,
            "context_view": "static_background",
            "positive_view": "static_holdout",
        }
        wide.append({**common, **{dataset: cells.get(dataset, "") for dataset in DATASETS}})
        for dataset in DATASETS:
            source = DATASET_TO_SOURCE[dataset]
            entry_rung = [key for key, _ in PLAN.SOURCES].index(source) + 1
            long_rows.append(
                {
                    **common,
                    "test_graph": dataset,
                    "auc": cells.get(dataset, ""),
                    "entry_rung": entry_rung,
                    "rel_to_entry": int(row["rung"]) - entry_rung,
                    "in_training": int(int(row["rung"]) >= entry_rung),
                    "eval_run": provenance.get(dataset, ""),
                }
            )
    return wide, long_rows, missing


def entry_diagnostics(rows: list[dict[str, object]]) -> None:
    cells = {
        (int(row["rung"]), str(row["test_graph"])): float(row["auc"])
        for row in rows if row["auc"] != ""
    }
    jumps = []
    for row in rows:
        rung = int(row["rung"])
        if row["auc"] == "" or int(row["entry_rung"]) != rung or rung == 1:
            continue
        before = cells.get((rung - 1, str(row["test_graph"])))
        if before is not None:
            jumps.append(float(row["auc"]) - before)
    if jumps:
        print(
            f"entry jumps: {sum(delta > 0 for delta in jumps)}/{len(jumps)} positive; "
            f"mean={sum(jumps)/len(jumps):+.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-root", type=Path,
        default=Path("/dataMeR1/phil/gfm/prodigy-nmlsplit-h2/log"),
    )
    parser.add_argument("--out-dir", type=Path, default=HERE / "data")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    wide, long_rows, missing = assemble(args.log_root)
    common = [
        "rung", "n_sources", "added", "sources", "model_prefix", "checkpoint_step",
        "n_hop", "hop_sizes", "node_limit", "nm_walk_hops", "context_view",
        "positive_view",
    ]
    write_csv(args.out_dir / "nm_ladder_train_test_nhop2.csv", common + DATASETS, wide)
    write_csv(
        args.out_dir / "nm_ladder_train_test_nhop2_long.csv",
        common + ["test_graph", "auc", "entry_rung", "rel_to_entry", "in_training", "eval_run"],
        long_rows,
    )
    entry_diagnostics(long_rows)
    if missing:
        print(f"{len(missing)} rungs have missing cells", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        return 0 if args.allow_partial else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
