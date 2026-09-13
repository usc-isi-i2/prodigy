#!/usr/bin/env python3
"""Validate and combine the matched three-seed flagship CLS ladders."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


ARMS = ("baseline", "objective", "exposure", "schedule", "composition")
SEEDS = (0, 1, 2)
RUNGS = tuple(range(1, 9))
TARGETS = (
    "covid_political",
    "election2020",
    "facebook_page_reference",
    "twibot20",
    "ukr_rus_suspended",
)
MODEL_RE = re.compile(
    r"^nmi_(?P<arm>baseline|objective|exposure|schedule|composition)_"
    r"r(?P<rung>[1-8])_s(?P<seed>[0-2])$"
)
EXPECTED = {
    (arm, rung, seed, target)
    for arm in ARMS for rung in RUNGS for seed in SEEDS for target in TARGETS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(args.input_root.glob("worker_*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            match = MODEL_RE.fullmatch(row["model_id"])
            if match is None:
                raise ValueError(f"unexpected model id in {path}:{line_number}: {row['model_id']}")
            key = (
                match["arm"], int(match["rung"]), int(match["seed"]), row["dataset"]
            )
            if int(row["training_seed"]) != key[2]:
                raise ValueError(f"training seed mismatch at {path}:{line_number}")
            row.update(arm=key[0], rung=key[1])
            rows.append((key, row))

    keys = [key for key, _ in rows]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate classification cells: {duplicates[:12]}")
    observed = set(keys)
    if observed != EXPECTED:
        missing = sorted(EXPECTED - observed)
        extra = sorted(observed - EXPECTED)
        raise ValueError(f"classification coverage mismatch: missing={missing[:12]} extra={extra[:12]}")

    fingerprints: dict[str, set[str]] = {target: set() for target in TARGETS}
    for _, row in rows:
        if int(row["episodes"]) != 128 or int(row["n_way"]) != 2 or int(row["n_shot"]) != 10:
            raise ValueError(f"bad classification protocol row: {row}")
        if int(row["checkpoint_step"]) <= 0:
            raise ValueError(f"invalid selected checkpoint: {row}")
        for metric in ("roc_auc", "accuracy", "f1"):
            value = float(row[metric])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid {metric}={value}: {row}")
        if not row.get("checkpoint_sha256") or not row.get("training_revision"):
            raise ValueError(f"missing checkpoint provenance: {row}")
        fingerprints[row["dataset"]].add(row["episode_fingerprint"])
    drift = {target: values for target, values in fingerprints.items() if len(values) != 1}
    if drift:
        raise ValueError(f"episode fingerprint drift: {drift}")

    fields = (
        "arm", "rung", "training_seed", "model_id", "sources", "checkpoint_step",
        "checkpoint", "checkpoint_sha256", "training_revision", "dataset", "episodes",
        "queries", "episode_fingerprint", "roc_auc", "accuracy", "f1",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for _, row in sorted(rows, key=lambda item: item[0]):
            out = {field: row[field] for field in fields}
            out["sources"] = ",".join(row["sources"])
            writer.writerow(out)
    print(f"wrote {len(rows)} validated classification cells to {args.output}")


if __name__ == "__main__":
    main()
