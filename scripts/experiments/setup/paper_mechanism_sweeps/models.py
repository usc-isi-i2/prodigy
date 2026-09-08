#!/usr/bin/env python3
"""Discover every fixed-step checkpoint from a completed mechanism sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.experiments.setup.paper_mechanism_sweeps.plan import (
    ARMS,
    CHECKPOINT_STEPS,
    SEEDS,
)
from scripts.experiments.setup.paper_three_seed.evaluate_fast_core import (
    normalize_sources,
    validate_terminal_status,
)


EXPECTED_JOBS = len(ARMS) * len(SEEDS)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def discover(run_dir: Path) -> list[dict]:
    validate_terminal_status(run_dir, EXPECTED_JOBS)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    registry = {f"paper_mech_{arm.name}": arm for arm in ARMS}
    observed_jobs = set()
    models = []
    for result_path in sorted(run_dir.glob("job_*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            raise ValueError(f"non-complete job under completed run: {result_path}")
        params = json.loads((result_path.parent / "effective_config.json").read_text(encoding="utf-8"))
        prefix = str(params.get("prefix", ""))
        if prefix not in registry:
            raise ValueError(f"unexpected mechanism prefix: {prefix}")
        arm = registry[prefix]
        seed = int(params["seed"])
        if seed not in SEEDS or not prefix.endswith(arm.name):
            raise ValueError(f"invalid arm/seed declaration: {prefix} seed={seed}")
        if float(params["neighbor_sampling_cross_source_prob"]) != arm.cross_graph_prob:
            raise ValueError(f"cross-graph probability drift for {prefix}")
        if int(params["emb_dim"]) != arm.emb_dim:
            raise ValueError(f"capacity drift for {prefix}")
        key = (arm.name, seed)
        if key in observed_jobs:
            raise ValueError(f"duplicate mechanism job: {key}")
        observed_jobs.add(key)
        checkpoint_dir = Path(result["checkpoint_dir"])
        for step in CHECKPOINT_STEPS:
            checkpoint = checkpoint_dir / f"state_dict_{step}.ckpt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            models.append(
                {
                    "model_id": f"paper_mech_{arm.name}_step{step}_s{seed}",
                    "family": "paper_mechanism",
                    "arm": arm.name,
                    "seed": seed,
                    "sources": normalize_sources(params["neighbor_sampling_source_subset"]),
                    "checkpoint": str(checkpoint),
                    "checkpoint_step": step,
                    "checkpoint_sha256": sha256(checkpoint),
                    "training_revision": manifest["revision"],
                    "training_exp_name": params["exp_name"],
                    "training_config": str(params["config"]),
                    "cross_graph_prob": arm.cross_graph_prob,
                    "emb_dim": arm.emb_dim,
                    "params": params,
                }
            )
    expected = {(arm.name, seed) for arm in ARMS for seed in SEEDS}
    if observed_jobs != expected:
        raise ValueError(
            f"mechanism job coverage mismatch: missing={sorted(expected-observed_jobs)} "
            f"extra={sorted(observed_jobs-expected)}"
        )
    ids = [row["model_id"] for row in models]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate mechanism model IDs")
    return models


def write_model_list(models: list[dict], output: Path, *, exclude_wide: bool = False) -> None:
    if exclude_wide:
        models = [row for row in models if row["emb_dim"] == 256]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(
            "model_id\tcheckpoint\tsources\ttraining_seed\tcheckpoint_step\t"
            "training_revision\tcheckpoint_sha256\n"
        )
        for row in sorted(models, key=lambda item: (item["arm"], item["seed"], item["checkpoint_step"])):
            handle.write(
                "\t".join(
                    (
                        row["model_id"],
                        row["checkpoint"],
                        ",".join(row["sources"]),
                        str(row["seed"]),
                        str(row["checkpoint_step"]),
                        row["training_revision"],
                        row["checkpoint_sha256"],
                    )
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-wide",
        action="store_true",
        help="Omit the 512-wide arm for evaluators whose architecture is fixed by one config.",
    )
    args = parser.parse_args()
    rows = discover(args.run_dir.resolve())
    selected = [row for row in rows if not args.exclude_wide or row["emb_dim"] == 256]
    write_model_list(rows, args.output.resolve(), exclude_wide=args.exclude_wide)
    print(f"wrote {len(selected)} verified fixed-step checkpoints to {args.output.resolve()}")


if __name__ == "__main__":
    main()
