#!/usr/bin/env python3
"""Validate and aggregate the complete Collab MLP hard-negative campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FINGERPRINT = "07f7af8e654bda27caad60ed74c479f48780343826543dd613d90cb4e979f9f4"
NEW_ARMS = ("cosine_hard8", "interaction_uniform", "interaction_hard8")
STRATA = ("seen_in_train", "novel_vs_train")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values: list[float]) -> dict[str, float | int]:
    return {"mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1)), "n": len(values)}


def load_new(root: Path) -> tuple[dict[str, list[dict]], set[str]]:
    grouped = {arm: [] for arm in NEW_ARMS}
    revisions = set()
    for arm in NEW_ARMS:
        for seed in range(3):
            directory = root / arm / f"seed{seed}"
            result = json.loads((directory / "results.json").read_text())
            if not result.get("complete") or result["arm"] != arm or result["seed"] != seed:
                raise ValueError(f"invalid cell identity: {directory}")
            if result["dataset_fingerprint"] != FINGERPRINT:
                raise ValueError(f"dataset mismatch: {directory}")
            selection_path = directory / "selection_frozen.json"
            if sha256_file(selection_path) != result["selection_frozen_sha256"]:
                raise ValueError(f"selection hash mismatch: {directory}")
            selection = json.loads(selection_path.read_text())["learned"]
            checkpoint = Path(selection["checkpoint"])
            if sha256_file(checkpoint) != selection["checkpoint_sha256"]:
                raise ValueError(f"checkpoint hash mismatch: {directory}")
            grouped[arm].append(result)
            revisions.add(result["revision"])
    return grouped, revisions


def load_baseline(root: Path) -> list[dict]:
    rows = []
    for seed in range(3):
        payload = json.loads((root / f"seed{seed}" / "results.json").read_text())
        if not payload.get("complete") or payload["seed"] != seed or payload["dataset_fingerprint"] != FINGERPRINT:
            raise ValueError(f"invalid baseline seed {seed}")
        row = next(value for value in payload["results"] if value["arm"] == "nonlinear_mlp_cosine")
        rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    grouped, revisions = load_new(args.root)
    if len(revisions) != 1:
        raise ValueError(f"new-cell revisions differ: {sorted(revisions)}")
    grouped = {"cosine_uniform": load_baseline(args.baseline_root), **grouped}
    metrics = {}
    for arm, rows in grouped.items():
        metrics[arm] = {
            "official_test_hits_at_50": summarize([row["official_test"]["hits_at_50"] for row in rows]),
            **{f"{stratum}_hits_at_50": summarize([row["test_strata"][stratum]["hits_at_50"] for row in rows]) for stratum in STRATA},
        }
    def mean(arm: str, key: str) -> float:
        return float(metrics[arm][key]["mean"])
    contrasts = {}
    for key in ("official_test_hits_at_50", "seen_in_train_hits_at_50", "novel_vs_train_hits_at_50"):
        contrasts[key] = {
            "hard_negative_effect_with_cosine": mean("cosine_hard8", key) - mean("cosine_uniform", key),
            "interaction_effect_with_uniform": mean("interaction_uniform", key) - mean("cosine_uniform", key),
            "interaction_effect_with_hard8": mean("interaction_hard8", key) - mean("cosine_hard8", key),
            "joint_effect_vs_baseline": mean("interaction_hard8", key) - mean("cosine_uniform", key),
        }
    output = {"complete": True, "expected_cells": 12, "reused_baseline_cells": 3,
              "new_cells": 9, "new_revision": next(iter(revisions)),
              "dataset_fingerprint": FINGERPRINT,
              "checks": {"all_expected_cells_present": True, "selection_hashes_match": True,
                         "checkpoint_hashes_match": True, "new_cell_revision_match": True,
                         "dataset_fingerprint_match": True},
              "metrics": metrics, "contrasts": contrasts}
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
