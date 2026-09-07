#!/usr/bin/env python3
"""Verify checkpoints and exact per-source episode identity across schedule arms."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from scripts.experiments.setup.final_core.core_plan import SOURCES as ALL_SOURCES
from scripts.experiments.setup.trace_schedule_scaling.make_plan import build_plan


def expand_schedule(arm):
    return [
        source
        for source, steps in zip(arm.segment_sources, arm.segment_steps)
        for _ in range(steps)
    ]


def find_audit(log_root: Path, run_name: str) -> Path:
    matches = [
        path for path in log_root.rglob("consumed_episodes.jsonl.gz")
        if run_name in path.as_posix()
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one consumed audit for {run_name}, got {matches}")
    return matches[0]


def read_audit(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def episode_payload(row: dict) -> dict:
    return {
        key: row[key]
        for key in (
            "anchor_ids", "member_ids", "query_roles", "context_node_counts",
            "source_ids", "anchor_sha256", "member_order_sha256", "member_set_sha256",
            "context_node_order_sha256", "context_edge_sha256",
        )
    }


def payload_digest(rows: list[dict]) -> str:
    payload = json.dumps(
        [episode_payload(row) for row in rows],
        separators=(",", ":"), sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_arm(arm, state_root: Path, log_root: Path, run_stamp: str, total_steps: int):
    run_name = f"tracesched_{arm.model_id}_{run_stamp}"
    checkpoint_root = state_root / run_name / "checkpoint"
    checkpoint = checkpoint_root / f"state_dict_{total_steps}.ckpt"
    sidecar = checkpoint_root / f"training_state_{total_steps}.ckpt"
    if not checkpoint.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"incomplete terminal checkpoint: {run_name}")
    training = torch.load(sidecar, map_location="cpu", weights_only=False)
    metadata = training.get("_training_checkpoint", {})
    if metadata.get("completed_steps") != total_steps:
        raise ValueError(f"wrong completed step for {run_name}")
    contract = metadata.get("parameter_contract", {})
    expected_contract = {
        "seed": arm.seed,
        "neighbor_sampling_source_subset": ",".join(arm.sources),
        "neighbor_sampling_source_schedule": ",".join(arm.segment_sources),
        "neighbor_sampling_source_schedule_steps": ",".join(
            str(value) for value in arm.segment_steps
        ),
        "neighbor_sampling_source_schedule_seed": 480100 + arm.seed,
        "neighbor_matching_member_policy": "uniform_shuffled",
        "neighbor_matching_member_seed": 380100 + arm.seed,
        "dataset_len_cap": total_steps,
        "batch_size": 1,
    }
    wrong = {
        key: (contract.get(key), value)
        for key, value in expected_contract.items()
        if contract.get(key) != value
    }
    if wrong:
        raise ValueError(f"training contract differs for {run_name}: {wrong}")

    audit_path = find_audit(log_root, run_name)
    rows = read_audit(audit_path)
    if len(rows) != total_steps or [row["step"] for row in rows] != list(
        range(1, total_steps + 1)
    ):
        raise ValueError(f"consumed audit is incomplete or out of order: {run_name}")
    expected_names = expand_schedule(arm)
    source_ids = {name: ALL_SOURCES.index(name) for name in arm.sources}
    for row, expected_name in zip(rows, expected_names):
        observed = set(int(value) for value in row["source_ids"])
        if observed != {source_ids[expected_name]}:
            raise ValueError(
                f"source schedule mismatch at {run_name} step {row['step']}: "
                f"expected {expected_name}, got {observed}"
            )
        if row["query_roles"] != [0, 0, 0, 1, 1, 1, 1]:
            raise ValueError(f"unexpected support/query role grid: {run_name}")
    by_source = {
        name: [
            row for row, expected_name in zip(rows, expected_names)
            if expected_name == name
        ]
        for name in arm.sources
    }
    return {
        "run_name": run_name,
        "checkpoint": str(checkpoint),
        "sidecar": str(sidecar),
        "audit": str(audit_path),
        "episodes": len(rows),
        "source_digests": {
            source: payload_digest(source_rows)
            for source, source_rows in by_source.items()
        },
        "source_rows": by_source,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--total-steps", type=int, default=2500)
    parser.add_argument("--replay-block", type=int, default=100)
    parser.add_argument("--rungs", default="2,3,4")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = build_plan(
        total_steps=args.total_steps,
        replay_block=args.replay_block,
        rungs=tuple(int(value) for value in args.rungs.split(",")),
        seeds=tuple(int(value) for value in args.seeds.split(",")),
    )
    receipts = []
    grouped = defaultdict(list)
    for arm in arms:
        receipt = verify_arm(
            arm, args.state_root, args.log_root, args.run_stamp, args.total_steps
        )
        grouped[(arm.rung, arm.seed)].append((arm, receipt))
        receipts.append({key: value for key, value in receipt.items() if key != "source_rows"})
    for key, group in grouped.items():
        reference_arm, reference = group[0]
        for arm, receipt in group[1:]:
            if receipt["source_digests"] != reference["source_digests"]:
                raise ValueError(f"per-source episode multiset differs in group {key}")
            for source in reference_arm.sources:
                left = [episode_payload(row) for row in reference["source_rows"][source]]
                right = [episode_payload(row) for row in receipt["source_rows"][source]]
                if left != right:
                    raise ValueError(
                        f"per-source episode order differs for {key}/{source}: "
                        f"{reference_arm.schedule} versus {arm.schedule}"
                    )
    # Retained sources also share private streams across mixture sizes. A
    # smaller-rung draw must be an exact prefix of the larger-rung draw, not a
    # separately resampled approximation to the same source distribution.
    by_seed_source = defaultdict(list)
    for (rung, seed), group in grouped.items():
        reference_arm, reference = group[0]
        for source in reference_arm.sources:
            by_seed_source[(seed, source)].append(
                (
                    rung,
                    [
                        episode_payload(row)
                        for row in reference["source_rows"][source]
                    ],
                )
            )
    for key, streams in by_seed_source.items():
        longest_rung, longest = max(streams, key=lambda item: len(item[1]))
        for rung, stream in streams:
            if longest[: len(stream)] != stream:
                raise ValueError(
                    f"cross-rung source prefix differs for {key}: "
                    f"rung {rung} is not a prefix of rung {longest_rung}"
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "complete",
                "arms": len(receipts),
                "total_steps": args.total_steps,
                "schedule_is_only_data_intervention": True,
                "cross_rung_source_prefixes_match": True,
                "receipts": receipts,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(
        f"OK: {len(receipts)} arms; exact per-source consumed episode sequences "
        "match across schedules"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
