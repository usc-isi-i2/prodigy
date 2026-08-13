#!/usr/bin/env python3
"""Assemble terminal sequential-ladder evals and pair them with the h2 control."""

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
SETUP = REPO_ROOT / "scripts/experiments/setup/nm_ladder_sequential_nhop2"


def load_plan_module():
    path = SETUP / "make_configs.py"
    spec = importlib.util.spec_from_file_location("nm_ladder_seq_make_configs", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import experiment plan from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLAN = load_plan_module()
DATASETS = list(PLAN.DATASET_KEYS)
KEY_OF_DATASET = {dataset: key for key, dataset, _ in PLAN.SOURCES}
CANONICAL = {key: canonical for key, _, canonical in PLAN.SOURCES}


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


def eval_row(log_root: Path, prefix: str) -> tuple[dict[str, float], dict[str, str]]:
    pattern = re.compile(rf"^eval_{re.escape(prefix)}_to_(?P<test>.+?)_nm_3shot_30way")
    values: dict[str, float] = {}
    provenance: dict[str, str] = {}
    run_dirs = sorted(
        (path for path in log_root.glob(f"eval_{prefix}_to_*_nm_3shot_30way*") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
    )
    for run_dir in run_dirs:
        match = pattern.match(run_dir.name)
        if match is None or match["test"] not in DATASETS:
            continue
        value = latest_roc_auc(run_dir)
        if value is not None:
            values[match["test"]] = value
            provenance[match["test"]] = run_dir.name
    return values, provenance


def assemble(log_root: Path):
    wide = []
    long_rows = []
    missing = []
    for row in PLAN.plan():
        rung = int(row["rung"])
        prefix = str(row["prefix"])
        values, provenance = eval_row(log_root, prefix)
        absent = [dataset for dataset in DATASETS if dataset not in values]
        if absent:
            missing.append(f"r{rung} -> {prefix}: {', '.join(absent)}")
        sources = list(row["sources"])
        wide.append(
            {
                "schedule": "sequential",
                "n_hop": 2,
                "hop_sizes": "9,9",
                "node_limit": 101,
                "nm_walk_hops": 1,
                "checkpoint_step": 40_000,
                "rung": rung,
                "n_sources": rung,
                "added": row["added"],
                "sources": " ".join(sources),
                "block_steps": " ".join(str(step) for step in row["steps"]),
                "model_prefix": prefix,
                **{dataset: values.get(dataset, "") for dataset in DATASETS},
            }
        )
        for dataset in DATASETS:
            key = KEY_OF_DATASET[dataset]
            entry_rung = [source[0] for source in PLAN.SOURCES].index(key) + 1
            long_rows.append(
                {
                    "schedule": "sequential",
                    "rung": rung,
                    "n_sources": rung,
                    "test_graph": dataset,
                    "test_canonical": CANONICAL[key],
                    "auc": values.get(dataset, ""),
                    "entry_rung": entry_rung,
                    "rel_to_entry": rung - entry_rung,
                    "in_training": int(rung >= entry_rung),
                    "is_newcomer": int(rung == entry_rung),
                    "added": row["added"],
                    "sources": " ".join(sources),
                    "block_steps": " ".join(str(step) for step in row["steps"]),
                    "model_prefix": prefix,
                    "eval_run": provenance.get(dataset, ""),
                }
            )
    return wide, long_rows, missing


def read_control(path: Path) -> dict[tuple[int, str], float]:
    if not path.is_file():
        return {}
    values = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("order", "A") != "A" or not row.get("auc"):
                continue
            values[(int(row["rung"]), row["test_graph"])] = float(row["auc"])
    return values


def pair_with_control(long_rows, control):
    paired = []
    for row in long_rows:
        if row["auc"] == "":
            continue
        key = (int(row["rung"]), str(row["test_graph"]))
        if key not in control:
            continue
        sequential = float(row["auc"])
        interleaved = control[key]
        if int(row["is_newcomer"]):
            role = "newcomer"
        elif int(row["in_training"]):
            role = "incumbent"
        else:
            role = "heldout"
        paired.append(
            {
                "rung": row["rung"],
                "test_graph": row["test_graph"],
                "entry_rung": row["entry_rung"],
                "rel_to_entry": row["rel_to_entry"],
                "role": role,
                "auc_interleaved": interleaved,
                "auc_sequential": sequential,
                "delta_sequential_minus_interleaved": sequential - interleaved,
            }
        )
    return paired


def write_csv(path: Path, fieldnames: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def print_diagnostics(paired) -> None:
    if not paired:
        print("paired control diagnostics unavailable")
        return
    for role in ("all", "newcomer", "incumbent", "heldout"):
        rows = paired if role == "all" else [row for row in paired if row["role"] == role]
        if not rows:
            continue
        deltas = [float(row["delta_sequential_minus_interleaved"]) for row in rows]
        print(
            f"{role:>9}: n={len(deltas):2d}, mean delta={sum(deltas) / len(deltas):+.4f}, "
            f"sequential wins={sum(delta > 0 for delta in deltas)}/{len(deltas)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-root", type=Path,
        default=Path("/dataMeR1/phil/gfm/prodigy-nmlh2seq/log"),
    )
    parser.add_argument("--out-dir", type=Path, default=HERE / "data")
    parser.add_argument(
        "--control-long", type=Path,
        default=REPO_ROOT / "scripts/experiments/analysis/transfer/ablations/prodigy_nm/context_depth/nm_ladder_nhop2/data/nm_ladder_nhop2_long.csv",
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    wide, long_rows, missing = assemble(args.log_root)
    wide_fields = [
        "schedule", "n_hop", "hop_sizes", "node_limit", "nm_walk_hops",
        "checkpoint_step", "rung", "n_sources", "added", "sources",
        "block_steps", "model_prefix", *DATASETS,
    ]
    long_fields = [
        "schedule", "rung", "n_sources", "test_graph", "test_canonical", "auc",
        "entry_rung", "rel_to_entry", "in_training", "is_newcomer", "added",
        "sources", "block_steps", "model_prefix", "eval_run",
    ]
    write_csv(args.out_dir / "nm_ladder_sequential_nhop2.csv", wide_fields, wide)
    write_csv(args.out_dir / "nm_ladder_sequential_nhop2_long.csv", long_fields, long_rows)

    control = read_control(args.control_long)
    paired = pair_with_control(long_rows, control)
    if control:
        write_csv(
            args.out_dir / "nm_ladder_schedule_comparison_long.csv",
            [
                "rung", "test_graph", "entry_rung", "rel_to_entry", "role",
                "auc_interleaved", "auc_sequential",
                "delta_sequential_minus_interleaved",
            ],
            paired,
        )
    else:
        print(f"WARN: no two-hop control table at {args.control_long}; comparison not written")
    print_diagnostics(paired)

    if missing:
        print(f"{len(missing)} rungs have missing test cells", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        if not args.allow_partial:
            print("exiting 1; pass --allow-partial to assemble incomplete results", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
