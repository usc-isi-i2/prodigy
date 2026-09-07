#!/usr/bin/env python3
"""Validate the complete shared-graph leave-one-source-out training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loo_plan import CHECKPOINT_STEP, build_models, checkpoint_path, physical_jobs


def verify(run_dir: Path) -> dict:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError(f"training status is {status.get('status')!r}, not complete")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("gpus") != [0, 1, 2, 3]:
        raise ValueError(f"expected physical GPUs 0--3, got {manifest.get('gpus')}")
    jobs = manifest.get("jobs", [])
    expected_models = {model.model_id: model for model in build_models()}
    if {row.get("prefix") for row in jobs} != set(expected_models):
        raise ValueError("manifest does not contain exactly the nine LOO prefixes")
    physical_gpus = set()
    steady_steps = 0
    steady_seconds = []
    exact_resume_supported = []
    for index, job in enumerate(physical_jobs()):
        row = next(row for row in jobs if row["prefix"] == job.model.model_id)
        checks = {
            "seed": 0,
            "neighbor_sampling_source_subset": ",".join(job.model.sources),
            "batch_size": 4,
            "dataset_len_cap": 2500,
            "epochs": 1,
            "neighbor_sampling_episode_source": "graph_id",
            "neighbor_sampling_episode_source_weighting": "balanced",
            "edge_view": "static_train",
            "target_edge_view": "static_test",
        }
        for key, wanted in checks.items():
            if row.get(key) != wanted:
                raise ValueError(
                    f"{job.model.model_id}: {key} expected {wanted!r}, got {row.get(key)!r}"
                )
        checkpoint = checkpoint_path(run_dir, job, "")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        training_state = checkpoint.with_name(
            checkpoint.name.replace("state_dict_", "training_state_")
        )
        if not training_state.is_file():
            raise FileNotFoundError(training_state)
        job_dir = run_dir / f"job_{index:03d}"
        result = json.loads((job_dir / "result.json").read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            raise ValueError(f"{job_dir}: status is not complete")
        physical_gpus.add(int(result["physical_gpu"]))
        steady_steps += int(result.get("steady_steps", 0))
        if result.get("steady_started") is not None and result.get("steady_finished") is not None:
            steady_seconds.append((float(result["steady_started"]), float(result["steady_finished"])))
        exact_resume_supported.append(bool(row.get("exact_resume_supported", False)))
    if physical_gpus != {0, 1, 2, 3}:
        raise ValueError(f"not every owned GPU was used: {sorted(physical_gpus)}")
    combined_window = (
        max(end for _, end in steady_seconds) - min(start for start, _ in steady_seconds)
        if steady_seconds else 0.0
    )
    return {
        "status": "complete",
        "loo_models": 9,
        "seed": 0,
        "checkpoint_step": CHECKPOINT_STEP,
        "heldout_test_cells_expected": 9,
        "physical_gpus": sorted(physical_gpus),
        "weight_and_training_state_checkpoints": True,
        "exact_resume_supported_for_all": all(exact_resume_supported),
        "steady_steps_summed": steady_steps,
        "combined_measurement_window_seconds": combined_window,
        "aggregate_optimizer_steps_per_second": (
            steady_steps / combined_window if combined_window else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
