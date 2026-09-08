#!/usr/bin/env python3
"""Validate the fixed-stream downstream classification mechanism grid."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

from scripts.experiments.setup.paper_mechanism_sweeps.plan import CHECKPOINT_STEPS, SEEDS


TARGETS = (
    "covid_political",
    "election2020",
    "facebook_page_reference",
    "twibot20",
    "ukr_rus_suspended",
)
ARM_NAMES = ("mix_p000", "mix_p010", "mix_p025", "mix_p050", "mix_p100")
MODEL_RE = re.compile(
    r"^paper_mech_(?P<arm>mix_p000|mix_p010|mix_p025|mix_p050|mix_p100)_"
    r"step(?P<step>2000|4000|6000|8000|10000)_s(?P<seed>[0-2])$"
)
EXPECTED = {
    (arm, step, seed, target)
    for arm in ARM_NAMES for step in CHECKPOINT_STEPS for seed in SEEDS for target in TARGETS
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input_root.glob("worker_*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            match = MODEL_RE.fullmatch(row["model_id"])
            if match is None:
                raise ValueError(f"unexpected model id in {path}:{line_number}: {row['model_id']}")
            key = (match["arm"], int(match["step"]), int(match["seed"]), row["dataset"])
            if int(row["training_seed"]) != key[2] or int(row["checkpoint_step"]) != key[1]:
                raise ValueError(f"seed/checkpoint mismatch at {path}:{line_number}")
            row.update(arm=key[0])
            rows.append((key, row))
    keys = [key for key, _ in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate mechanism classification cells")
    observed = set(keys)
    if observed != EXPECTED:
        raise ValueError(
            f"classification coverage mismatch: missing={len(EXPECTED-observed)} "
            f"extra={len(observed-EXPECTED)}"
        )
    fingerprints = {target: set() for target in TARGETS}
    for _, row in rows:
        if int(row["episodes"]) != 128 or int(row["n_way"]) != 2 or int(row["n_shot"]) != 10:
            raise ValueError(f"classification protocol drift: {row}")
        for metric in ("roc_auc", "accuracy", "f1"):
            value = float(row[metric])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid {metric}={value}")
        if not row.get("checkpoint_sha256") or not row.get("training_revision"):
            raise ValueError("missing classification checkpoint provenance")
        fingerprints[row["dataset"]].add(row["episode_fingerprint"])
    drift = {target: values for target, values in fingerprints.items() if len(values) != 1}
    if drift:
        raise ValueError(f"classification episode fingerprint drift: {drift}")
    fields = (
        "arm", "training_seed", "model_id", "sources", "checkpoint_step", "checkpoint",
        "checkpoint_sha256", "training_revision", "dataset", "episodes", "queries",
        "episode_fingerprint", "roc_auc", "accuracy", "f1",
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
