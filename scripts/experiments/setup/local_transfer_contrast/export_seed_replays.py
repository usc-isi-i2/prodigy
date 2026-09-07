"""Convert independent-checkpoint replays into the audited health-analysis format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.run_shared_graph import write_json
from scripts.experiments.setup.target_performance_mechanisms.analyze_mixture_predictions import input_labels


RAW_STAGES = ("raw_center", "raw_context", "raw_joint")


def reference_for(roots, target):
    matches = [root / target for root in roots if (root / target / "cache.json").is_file()]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one reference cache for {target}, got {matches}")
    return matches[0]


def assert_same_cache(actual, reference):
    for key in ("episode_fingerprint", "batch_sha256", "episodes", "graph_path"):
        if actual[key] != reference[key]:
            raise ValueError(f"fixed-input contract differs at {key}")


def full_model_rows(directory):
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    selected = {
        row["model_id"]: row for row in rows
        if row["variant"] == "baseline" and row["decoder"] == "full_model"
    }
    if len(selected) != 9:
        raise ValueError(f"expected nine singleton full-model rows, got {len(selected)}")
    return selected


def load_stream(root, target, stream, seed, reference_roots, raw_parity_atol):
    directory = root / target
    labels = input_labels(directory, target)
    reference = input_labels(reference_for(reference_roots, target), target)
    assert_same_cache(labels["cache"], reference["cache"])
    rows = full_model_rows(directory)
    model_ids = sorted(rows)
    batches = [
        torch.load(directory / "batches" / f"batch_{index:03d}.pt", map_location="cpu", weights_only=False)
        for index in range(32)
    ]
    query_masks = [batch[5].reshape(-1, 2)[:, 0].bool() for batch in batches]
    output = {
        "target": target,
        "stream": stream,
        "labels": labels,
        "models": {},
        "input": {"embeddings": {}},
        "receipts": {"seed": seed, "root": str(directory), **labels["cache"]},
    }
    reference_raw = None
    for model_id in model_ids:
        row = rows[model_id]
        if row["sources"] != [model_id.removeprefix("ss_")] or f"_s{seed}_" not in row["checkpoint"]:
            raise ValueError(f"checkpoint/source identity mismatch for {model_id}")
        records = torch.load(directory / f"{model_id}__baseline.pt", map_location="cpu", weights_only=False)
        if len(records) != 32:
            raise ValueError(f"incomplete replay for {model_id}")
        for index, record in enumerate(records):
            if record["batch"] != index or record["batch_sha256"] != labels["cache"]["batch_sha256"][index]:
                raise ValueError(f"batch receipt mismatch for {model_id}/{index}")
        logits = {name: torch.cat([record["logits"][name] for record in records])
                  for name in records[0]["logits"]}
        raw = {stage: torch.cat([
            record["embeddings"][stage][mask] for record, mask in zip(records, query_masks)
        ]) for stage in RAW_STAGES}
        if reference_raw is None:
            reference_raw = raw
            output["input"]["embeddings"] = raw
        else:
            for stage in RAW_STAGES:
                torch.testing.assert_close(
                    reference_raw[stage], raw[stage], rtol=0, atol=raw_parity_atol,
                )
        output["models"][model_id] = {
            "weights_sha256": row["weights_sha256"], "logits": logits,
        }
    n = len(labels["local_y"])
    if any(len(values) != n for values in output["input"]["embeddings"].values()):
        raise ValueError("raw query embedding count mismatch")
    if any(len(model["logits"]["full_model"]) != n for model in output["models"].values()):
        raise ValueError("query logit count mismatch")
    return output


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--fresh-root", type=Path, required=True)
    parser.add_argument("--reference-original-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-fresh-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--raw-parity-atol", type=float, default=0.0,
                        help="Tolerance for CUDA-reduced raw observer embeddings; input tensors stay hash-exact.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a new output directory")
    if not 0 <= args.raw_parity_atol <= 1e-6:
        raise ValueError("raw parity tolerance must be in [0, 1e-6]")
    targets = sorted(path.name for path in args.original_root.iterdir() if (path / "cache.json").is_file())
    if len(targets) != 5:
        raise ValueError(f"expected five complete target directories, got {targets}")
    args.output.mkdir(parents=True)
    receipts = []
    for target in targets:
        target_out = args.output / target
        target_out.mkdir()
        for stream, root, references in (
            ("original", args.original_root, args.reference_original_roots),
            ("fresh", args.fresh_root, args.reference_fresh_roots),
        ):
            record = load_stream(root, target, stream, args.seed, references, args.raw_parity_atol)
            destination = target_out / f"{stream}.pt"
            torch.save(record, destination)
            receipts.append({"target": target, "stream": stream, "queries": len(record["labels"]["local_y"]),
                             "episode_fingerprint": record["receipts"]["episode_fingerprint"]})
            print(json.dumps(receipts[-1]), flush=True)
    write_json(args.output / "protocol.json", {
        "checkpoint_seed": args.seed,
        "training_seed_isolated": True,
        "target_episode_seed": 0,
        "fresh_episode_offset": 100003,
        "models": 9,
        "targets": targets,
        "streams": ["original", "fresh"],
        "fixed_episode_identity_verified_against_seed0": True,
        "raw_observer_parity_atol": args.raw_parity_atol,
        "receipts": receipts,
    })
    write_json(args.output / "DONE.json", {"complete": True, "cells": 90, "files": 10})


if __name__ == "__main__":
    main()
