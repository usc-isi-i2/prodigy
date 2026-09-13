#!/usr/bin/env python3
"""Build the frozen Stage-A early-transfer evaluation manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
FINAL_CORE = HERE.parent / "final_core"
sys.path.insert(0, str(FINAL_CORE))

from core_plan import SOURCES  # noqa: E402


PROTOCOL_ID = "pilot_transfer_selection_v1"
PILOT_STEPS = (100, 300, 900)
SEEDS = (0, 1, 2)
EPISODES = 512
TRAINING_RUN_STAMP = "20260807"
DEFAULT_STATE_ROOT = Path(
    "/dataMeR1/phil/gfm/worktree-runtime-archive-20260812/"
    "prodigy-final-core/files/state/final_core"
)
FIELDNAMES = (
    "cell_id",
    "checkpoint_job_id",
    "protocol_id",
    "model_id",
    "sources",
    "training_seed",
    "checkpoint_step",
    "checkpoint",
    "target",
    "evaluation_split",
    "message_passing_view",
    "target_edge_view",
    "episodes",
    "primary_metric",
    "protocol_sha256",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def checkpoint_path(
    state_root: Path, source: str, seed: int, step: int, run_stamp: str
) -> Path:
    return (
        state_root
        / f"finalcore_ss_{source}_s{seed}_{run_stamp}"
        / "checkpoint"
        / f"state_dict_{step}.ckpt"
    )


def build_rows(
    state_root: Path = DEFAULT_STATE_ROOT,
    run_stamp: str = TRAINING_RUN_STAMP,
    protocol_path: Path = HERE / "protocol.yaml",
) -> list[dict[str, str | int]]:
    protocol_hash = file_sha256(protocol_path)
    rows: list[dict[str, str | int]] = []
    for source in SOURCES:
        model_id = f"ss_{source}"
        for seed in SEEDS:
            for step in PILOT_STEPS:
                job_id = f"{model_id}|seed={seed}|step={step}"
                checkpoint = checkpoint_path(state_root, source, seed, step, run_stamp)
                for target in SOURCES:
                    rows.append(
                        {
                            "cell_id": f"{job_id}|target={target}|split=validation",
                            "checkpoint_job_id": job_id,
                            "protocol_id": PROTOCOL_ID,
                            "model_id": model_id,
                            "sources": source,
                            "training_seed": seed,
                            "checkpoint_step": step,
                            "checkpoint": str(checkpoint),
                            "target": target,
                            "evaluation_split": "validation",
                            "message_passing_view": "static_train",
                            "target_edge_view": "static_validation",
                            "episodes": EPISODES,
                            "primary_metric": "neighbor_matching_accuracy",
                            "protocol_sha256": protocol_hash,
                        }
                    )
    validate_rows(rows)
    return rows


def validate_rows(rows: list[dict[str, str | int]]) -> None:
    expected_cells = len(SOURCES) * len(SEEDS) * len(PILOT_STEPS) * len(SOURCES)
    if len(rows) != expected_cells:
        raise ValueError(f"expected {expected_cells} cells, found {len(rows)}")
    cell_ids = [str(row["cell_id"]) for row in rows]
    if len(cell_ids) != len(set(cell_ids)):
        raise ValueError("manifest contains duplicate cell IDs")
    jobs = {str(row["checkpoint_job_id"]) for row in rows}
    if len(jobs) != len(SOURCES) * len(SEEDS) * len(PILOT_STEPS):
        raise ValueError(f"expected 81 checkpoint jobs, found {len(jobs)}")
    expected_targets = set(SOURCES)
    for job_id in jobs:
        job_rows = [row for row in rows if row["checkpoint_job_id"] == job_id]
        targets = {str(row["target"]) for row in job_rows}
        if len(job_rows) != len(SOURCES) or targets != expected_targets:
            raise ValueError(f"job {job_id} does not contain the complete target panel")
    for key, expected in (
        ("evaluation_split", {"validation"}),
        ("message_passing_view", {"static_train"}),
        ("target_edge_view", {"static_validation"}),
        ("episodes", {EPISODES}),
        ("checkpoint_step", set(PILOT_STEPS)),
        ("training_seed", set(SEEDS)),
    ):
        observed = {row[key] for row in rows}
        if observed != expected:
            raise ValueError(f"unexpected {key}: {observed!r}; expected {expected!r}")


def write_tsv(rows: list[dict[str, str | int]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=FIELDNAMES, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_jobs(
    rows: list[dict[str, str | int]], step: int
) -> list[dict[str, object]]:
    if step not in PILOT_STEPS:
        raise ValueError(f"unsupported pilot step {step}")
    jobs: dict[str, dict[str, object]] = {}
    for row in rows:
        if int(row["checkpoint_step"]) != step:
            continue
        job_id = str(row["checkpoint_job_id"])
        candidate = {
            "model_id": row["model_id"],
            "seed": row["training_seed"],
            "sources": [row["sources"]],
            "aliases": [f"pilot:{row['sources']}:step{step}"],
            "checkpoint": row["checkpoint"],
        }
        if job_id in jobs and jobs[job_id] != candidate:
            raise ValueError(f"inconsistent checkpoint job {job_id}")
        jobs[job_id] = candidate
    output = [jobs[key] for key in sorted(jobs)]
    if len(output) != len(SOURCES) * len(SEEDS):
        raise ValueError(f"step {step}: expected 27 checkpoint jobs, found {len(output)}")
    return output


def write_job_manifests(
    rows: list[dict[str, str | int]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for step in PILOT_STEPS:
        path = output_dir / f"jobs_step{step}.json"
        path.write_text(
            json.dumps(checkpoint_jobs(rows, step), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--training-run-stamp", default=TRAINING_RUN_STAMP)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--job-manifest-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_rows(args.state_root, args.training_run_stamp)
    if args.output:
        write_tsv(rows, args.output)
    if args.job_manifest_dir:
        write_job_manifests(rows, args.job_manifest_dir)
    print(
        f"protocol={PROTOCOL_ID} cells={len(rows)} "
        f"checkpoint_jobs={len({row['checkpoint_job_id'] for row in rows})} "
        f"sources={len(SOURCES)} targets={len(SOURCES)} seeds={len(SEEDS)} "
        f"steps={','.join(map(str, PILOT_STEPS))} output={args.output or '-'} "
        f"job_manifest_dir={args.job_manifest_dir or '-'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
