#!/usr/bin/env python3
"""Convert schedule replays into the audited TRACE health-analysis format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.run_shared_graph import write_json
from scripts.experiments.setup.target_performance_mechanisms.analyze_mixture_predictions import input_labels
from scripts.experiments.setup.trace_schedule_scaling.make_plan import build_plan


RAW_STAGES = ("raw_center", "raw_context", "raw_joint")
TARGETS = (
    "election2020", "ukr_rus_suspended", "twibot20", "cp_hk",
    "facebook_page_reference",
)


def support_loo_prototype_accuracy(embeddings, batch):
    """Repeat each episode's support-only U1 competence over its query rows."""
    labels = batch[2].argmax(1)
    query = batch[5].reshape(-1, 2)[:, 0].bool()
    tasks = batch[0].task_id_per_sample
    x = F.normalize(embeddings, dim=1)
    result = embeddings.new_empty(len(labels))
    for task in tasks.unique(sorted=True):
        support_indices = torch.where((tasks == task) & ~query)[0]
        correct = []
        for held_out in support_indices:
            keep = support_indices[support_indices != held_out]
            prototypes = torch.stack(
                [
                    x[keep][labels[keep] == class_id].mean(0)
                    for class_id in range(2)
                ]
            )
            prediction = int(
                (x[held_out] @ F.normalize(prototypes, dim=1).T).argmax()
            )
            correct.append(prediction == int(labels[held_out]))
        result[(tasks == task) & query] = sum(correct) / len(correct)
    return result[query]


def full_model_rows(directory: Path, model_ids: set[str]) -> dict[str, dict]:
    rows = [
        json.loads(line)
        for line in (directory / "metrics.jsonl").read_text().splitlines()
    ]
    selected = {
        row["model_id"]: row
        for row in rows
        if row["variant"] == "baseline" and row["decoder"] == "full_model"
    }
    if set(selected) != model_ids:
        raise ValueError(
            f"full-model replay set differs in {directory}: "
            f"expected {len(model_ids)}, got {len(selected)}"
        )
    return selected


def load_stream(
    directory: Path, target: str, stream: str,
    arms_by_id: dict[str, object], raw_parity_atol: float,
) -> dict:
    labels = input_labels(directory, target)
    rows = full_model_rows(directory, set(arms_by_id))
    batches = [
        torch.load(
            directory / "batches" / f"batch_{index:03d}.pt",
            map_location="cpu", weights_only=False,
        )
        for index in range(32)
    ]
    query_masks = [batch[5].reshape(-1, 2)[:, 0].bool() for batch in batches]
    output = {
        "target": target,
        "stream": stream,
        "labels": labels,
        "models": {},
        "input": {"embeddings": {}},
        "receipts": {"root": str(directory), **labels["cache"]},
    }
    reference_raw = None
    for model_id in sorted(arms_by_id):
        arm = arms_by_id[model_id]
        row = rows[model_id]
        if tuple(row["sources"]) != arm.sources:
            raise ValueError(f"source identity mismatch for {model_id}")
        records = torch.load(
            directory / f"{model_id}__baseline.pt",
            map_location="cpu", weights_only=False,
        )
        if len(records) != 32:
            raise ValueError(f"incomplete replay for {target}/{stream}/{model_id}")
        for index, record in enumerate(records):
            if (
                record["batch"] != index
                or record["batch_sha256"] != labels["cache"]["batch_sha256"][index]
                or "embeddings" not in record
            ):
                raise ValueError(f"batch/export receipt mismatch for {model_id}/{index}")
        logits = {
            name: torch.cat([record["logits"][name] for record in records])
            for name in records[0]["logits"]
        }
        support_health = torch.cat(
            [
                support_loo_prototype_accuracy(
                    record["embeddings"]["U1_pre_meta"], batch
                )
                for record, batch in zip(records, batches)
            ]
        )
        raw = {
            stage: torch.cat(
                [
                    record["embeddings"][stage][mask]
                    for record, mask in zip(records, query_masks)
                ]
            )
            for stage in RAW_STAGES
        }
        if reference_raw is None:
            reference_raw = raw
            output["input"]["embeddings"] = raw
        else:
            for stage in RAW_STAGES:
                torch.testing.assert_close(
                    reference_raw[stage], raw[stage],
                    rtol=0, atol=raw_parity_atol,
                )
        output["models"][model_id] = {
            "rung": arm.rung,
            "seed": arm.seed,
            "schedule": arm.schedule,
            "sources": list(arm.sources),
            "weights_sha256": row["weights_sha256"],
            "logits": logits,
            "support_health": {
                "u1_loo_prototype_accuracy": support_health,
            },
        }
    query_count = len(labels["local_y"])
    if any(
        len(model["logits"]["full_model"]) != query_count
        for model in output["models"].values()
    ):
        raise ValueError("query logit count mismatch")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-parity-atol", type=float, default=1e-6)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("choose a new output directory")
    arms = build_plan()
    arms_by_id = {arm.model_id: arm for arm in arms}
    args.output.mkdir(parents=True)
    receipts = []
    fingerprints = {}
    for target in TARGETS:
        target_output = args.output / target
        target_output.mkdir()
        for stream in ("original", "fresh"):
            job_root = args.replay_root / stream / target
            directory = job_root / target
            if not (job_root / "DONE").is_file():
                raise ValueError(f"incomplete replay job: {job_root}")
            record = load_stream(
                directory, target, stream, arms_by_id, args.raw_parity_atol
            )
            torch.save(record, target_output / f"{stream}.pt")
            fingerprint = record["receipts"]["episode_fingerprint"]
            fingerprints[(target, stream)] = fingerprint
            receipts.append(
                {
                    "target": target,
                    "stream": stream,
                    "queries": len(record["labels"]["local_y"]),
                    "episode_fingerprint": fingerprint,
                }
            )
        if fingerprints[(target, "original")] == fingerprints[(target, "fresh")]:
            raise ValueError(f"original/fresh streams are not distinct for {target}")
    write_json(
        args.output / "protocol.json",
        {
            "models": len(arms),
            "targets": list(TARGETS),
            "streams": ["original", "fresh"],
            "original_episode_offset": 0,
            "fresh_episode_offset": 100003,
            "support_health": "U1 leave-one-support-out prototype accuracy",
            "raw_parity_atol": args.raw_parity_atol,
            "receipts": receipts,
        },
    )
    write_json(
        args.output / "DONE.json",
        {"complete": True, "cells": len(arms) * len(TARGETS) * 2},
    )
    print(f"OK: exported {len(arms)} models x {len(TARGETS)} targets x 2 streams")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
