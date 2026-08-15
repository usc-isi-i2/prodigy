#!/usr/bin/env python3
"""Validate and aggregate the two cross-task ladder evaluation sweeps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from scripts.experiments.setup.final_core.core_plan import ORDERS, SOURCES, build_models


DOWNSTREAM_TARGETS = (
    "covid_political",
    "election2020",
    "ukr_rus_suspended",
    "twibot20",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nm-results-root", required=True, type=Path)
    parser.add_argument("--downstream-worker-results", required=True, nargs="+", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ladder_models() -> dict[str, object]:
    return {
        model_for_ladder(order, rung).model_id: model_for_ladder(order, rung)
        for order in ORDERS
        for rung in range(1, 10)
    }


def model_for_ladder(order: str, rung: int):
    wanted = frozenset(ORDERS[order][:rung])
    matches = [model for model in build_models() if frozenset(model.sources) == wanted]
    if len(matches) != 1:
        raise AssertionError(f"expected one physical model for {order}/rung {rung}")
    return matches[0]


def load_nm(root: Path) -> dict[tuple[str, str], dict]:
    rows = {}
    for path in sorted(root.glob("seed_0/*/*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        key = row["model_id"], row["target"]
        if key in rows:
            raise ValueError(f"duplicate NM cell {key}")
        if row.get("protocol") != "fixed_nm_512_static_test_on_static_train_step100_v1":
            raise ValueError(f"wrong NM protocol in {path}")
        if int(row.get("checkpoint_step", -1)) != 100 or int(row.get("seed", -1)) != 0:
            raise ValueError(f"wrong NM step/seed in {path}")
        rows[key] = row
    expected = {(model_id, target) for model_id in ladder_models() for target in SOURCES}
    if set(rows) != expected:
        raise ValueError(f"NM grid mismatch: missing={sorted(expected-set(rows))[:5]} extra={sorted(set(rows)-expected)[:5]}")
    return rows


def load_downstream(paths: list[Path]) -> dict[tuple[int, str, str], dict]:
    rows = {}
    target_fingerprints: dict[str, set[str]] = {target: set() for target in DOWNSTREAM_TARGETS}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = int(row["training_seed"]), row["model_id"], row["dataset"]
            if key in rows:
                raise ValueError(f"duplicate downstream cell {key}")
            if int(row.get("checkpoint_step", -1)) != 2500:
                raise ValueError(f"wrong downstream checkpoint step in {path}")
            target_fingerprints[row["dataset"]].add(row["episode_fingerprint"])
            rows[key] = row
    expected = {
        (seed, model_id, target)
        for seed in (0, 1, 2)
        for model_id in ladder_models()
        for target in DOWNSTREAM_TARGETS
    }
    if set(rows) != expected:
        raise ValueError(
            f"downstream grid mismatch: missing={sorted(expected-set(rows))[:5]} "
            f"extra={sorted(set(rows)-expected)[:5]}"
        )
    drift = {target: values for target, values in target_fingerprints.items() if len(values) != 1}
    if drift:
        raise ValueError(f"downstream episode fingerprint drift: {drift}")
    return rows


def write_ladder_csv(path: Path, *, task: str, lookup: dict) -> None:
    fields = [
        "task", "order", "rung", "added_graph", "training_seed", "model_id",
        "target", "target_in_train", "checkpoint_step", "accuracy", "f1", "roc_auc",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        seeds = (0,) if task == "neighbor_matching" else (0, 1, 2)
        targets = SOURCES if task == "neighbor_matching" else DOWNSTREAM_TARGETS
        for seed in seeds:
            for order, sequence in ORDERS.items():
                for rung in range(1, 10):
                    model = model_for_ladder(order, rung)
                    for target in targets:
                        raw = (
                            lookup[model.model_id, target]
                            if task == "neighbor_matching"
                            else lookup[seed, model.model_id, target]
                        )
                        writer.writerow({
                            "task": task,
                            "order": order,
                            "rung": rung,
                            "added_graph": sequence[rung - 1],
                            "training_seed": seed,
                            "model_id": model.model_id,
                            "target": target,
                            "target_in_train": int(target in sequence[:rung]),
                            "checkpoint_step": 100 if task == "neighbor_matching" else 2500,
                            "accuracy": raw["accuracy"],
                            "f1": raw.get("f1_macro", raw.get("f1")),
                            "roc_auc": raw.get("roc_auc_ovr_macro", raw.get("roc_auc")),
                        })


def main() -> int:
    args = parse_args()
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    nm = load_nm(args.nm_results_root)
    downstream = load_downstream(args.downstream_worker_results)
    write_ladder_csv(output / "nm_step100_ladder.csv", task="neighbor_matching", lookup=nm)
    write_ladder_csv(
        output / "downstream_step2500_ladder.csv",
        task="classification",
        lookup=downstream,
    )
    manifest = {
        "nm_physical_cells": len(nm),
        "nm_logical_ladder_cells": 3 * 9 * 9,
        "downstream_physical_cells": len(downstream),
        "downstream_logical_ladder_cells": 3 * 3 * 9 * 4,
        "downstream_targets": list(DOWNSTREAM_TARGETS),
        "source_files": {
            str(path): fingerprint(path) for path in args.downstream_worker_results
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
