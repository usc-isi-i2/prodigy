#!/usr/bin/env python3
"""Validate and aggregate the frozen three-seed ogbl-collab MLP campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EXPECTED_SEEDS = tuple(range(3))
CAMPAIGN_ARMS = {
    "cosine": ("raw_cosine", "linear_cosine", "nonlinear_mlp_cosine"),
    "concat_sym": ("concat_linear_sym", "concat_mlp_sym"),
    "concat_ordered": ("concat_linear_ordered", "concat_mlp_ordered"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--campaign", choices=tuple(CAMPAIGN_ARMS), default="cosine")
    args = parser.parse_args()
    arms = CAMPAIGN_ARMS[args.campaign]

    missing: list[int] = []
    payloads = []
    for seed in EXPECTED_SEEDS:
        result_path = args.root / f"seed{seed}" / "results.json"
        if not result_path.exists():
            missing.append(seed)
            continue
        payload = json.loads(result_path.read_text())
        if not payload.get("complete") or payload.get("seed") != seed:
            raise ValueError(f"invalid result payload for seed {seed}: {result_path}")
        if [row["arm"] for row in payload["results"]] != list(arms):
            raise ValueError(f"arm mismatch for seed {seed}")
        payloads.append(payload)
    if missing:
        raise SystemExit(f"campaign incomplete; missing seeds: {missing}")

    revisions = {payload["revision"] for payload in payloads}
    fingerprints = {payload["dataset_fingerprint"] for payload in payloads}
    if len(revisions) != 1 or len(fingerprints) != 1:
        raise ValueError(
            f"incompatible campaign provenance: revisions={revisions}, fingerprints={fingerprints}"
        )

    aggregate = {
        "complete": True,
        "seeds": list(EXPECTED_SEEDS),
        "revision": next(iter(revisions)),
        "dataset_fingerprint": next(iter(fingerprints)),
        "campaign": args.campaign,
        "metrics": {},
    }
    for arm in arms:
        rows = [next(row for row in payload["results"] if row["arm"] == arm) for payload in payloads]
        if arm == "raw_cosine":
            reference = rows[0]
            if any(row != reference for row in rows[1:]):
                raise ValueError("deterministic raw baseline differs across seed outputs")
            values = [reference["official_test"]]
        else:
            values = [row["official_test"] for row in rows]
        aggregate["metrics"][arm] = {
            key: {
                "mean": float(np.mean([row[key] for row in values])),
                "sample_std": float(np.std([row[key] for row in values], ddof=1))
                if len(values) > 1
                else None,
                "n": len(values),
            }
            for key in values[0]
        }
    text = json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
