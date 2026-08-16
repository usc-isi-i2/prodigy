#!/usr/bin/env python3
"""Validate and combine the three-seed final-core classification ladder sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SEEDS = (0, 1, 2)
TARGETS = (
    "covid_political",
    "election2020",
    "facebook_page_reference",
    "twibot20",
    "ukr_rus_suspended",
)
MODELS = (
    "ss_ukr_rus",
    "ss_ukr_rus_suspended",
    "ss_twibot20",
    *(f"ord{order}_r{rung}" for order in "ABC" for rung in range(2, 9)),
    "all9",
)
EXPECTED_KEYS = {(seed, model, target) for seed in SEEDS for model in MODELS for target in TARGETS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    for path in sorted(args.input_root.glob("seed*_gpu*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error

    keys = [(int(row["training_seed"]), row["model_id"], row["dataset"]) for row in rows]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate cells: {duplicates[:5]}")
    observed = set(keys)
    if observed != EXPECTED_KEYS:
        missing = sorted(EXPECTED_KEYS - observed)
        extra = sorted(observed - EXPECTED_KEYS)
        raise ValueError(f"coverage mismatch: missing={missing[:10]} extra={extra[:10]}")

    fingerprints: dict[str, set[str]] = {target: set() for target in TARGETS}
    for row in rows:
        if int(row["checkpoint_step"]) != 2500 or int(row["episodes"]) != 128:
            raise ValueError(f"bad protocol row: {row}")
        for metric in ("roc_auc", "accuracy", "f1"):
            value = float(row[metric])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid {metric}={value}")
        fingerprints[row["dataset"]].add(row["episode_fingerprint"])
    drift = {target: values for target, values in fingerprints.items() if len(values) != 1}
    if drift:
        raise ValueError(f"episode fingerprint drift: {drift}")

    fields = (
        "training_seed", "model_id", "sources", "checkpoint_step", "dataset",
        "episodes", "queries", "episode_fingerprint", "roc_auc", "accuracy", "f1",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["training_seed"], item["model_id"], item["dataset"])):
            out = {field: row[field] for field in fields}
            out["sources"] = ",".join(row["sources"])
            writer.writerow(out)
    print(f"wrote {len(rows)} validated cells to {args.output}")


if __name__ == "__main__":
    main()
