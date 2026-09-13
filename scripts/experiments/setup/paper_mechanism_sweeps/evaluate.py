#!/usr/bin/env python3
"""Evaluate every fixed-step mechanism checkpoint on all nine NM receivers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[4]
from scripts.experiments.setup.nm_interventions_overnight.plan import TARGETS
from scripts.experiments.setup.paper_mechanism_sweeps.models import discover
from scripts.experiments.setup.paper_mechanism_sweeps.plan import ARMS, CHECKPOINT_STEPS, SEEDS
from scripts.experiments.setup.paper_three_seed.evaluate_fast_core import audit_cells, worker


PROTOCOL = "paper_mechanism_fixed_nm_v1"
EXPECTED_MODELS = len(ARMS) * len(SEEDS) * len(CHECKPOINT_STEPS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=ROOT / "scripts/experiments/setup/final_core/training.yaml",
    )
    parser.add_argument("--gpus", nargs="+", type=int, choices=range(4), default=[0, 1, 2, 3])
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers_per_gpu < 1 or args.episodes < 1:
        parser.error("worker and episode counts must be positive")
    jobs = discover(args.run_dir.resolve())
    if len(jobs) != EXPECTED_MODELS:
        raise ValueError(f"mechanism model coverage mismatch: {len(jobs)} != {EXPECTED_MODELS}")
    print(f"{len(jobs)} models x {len(TARGETS)} targets = {len(jobs) * len(TARGETS)} cells")
    if args.dry_run:
        return

    import torch
    from experiments.nm_campaign import atomic_json
    from experiments.params import get_params
    from experiments.run_shared_graph import prepare_shared_dataset, start_on_gpu
    from experiments.run_single_experiment import load_dataset

    args.output.mkdir(parents=True, exist_ok=True)
    invocation_id = f"{time.time_ns()}_{os.getpid()}"
    eval_params = get_params(["--config", str(args.eval_config.resolve())])
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    atomic_json(
        args.output / "plan.json",
        {
            "invocation_id": invocation_id,
            "models": [job["model_id"] for job in jobs],
            "targets": list(TARGETS),
            "episodes": args.episodes,
            "gpus": args.gpus,
            "workers_per_gpu": args.workers_per_gpu,
            "evaluation_revision": revision,
        },
    )
    started = time.time()
    atomic_json(args.output / "status.json", {"status": "running", "started": started,
                                                 "invocation_id": invocation_id})
    torch.set_num_threads(4)
    torch.autograd.set_detect_anomaly(False)
    dataset = prepare_shared_dataset(load_dataset(eval_params))
    missing = sorted(set(TARGETS) - set(dataset.graph.source_graph_names))
    if missing:
        raise ValueError(f"evaluation graph lacks registered targets: {missing}")
    slots = [gpu for gpu in args.gpus for _ in range(args.workers_per_gpu)]
    context = torch.multiprocessing.get_context("spawn")
    processes = []
    try:
        for index, gpu in enumerate(slots):
            selected = jobs[index::len(slots)]
            if not selected:
                continue
            process = context.Process(
                target=worker,
                args=(dataset, selected, eval_params, str(args.output), index,
                      args.episodes, invocation_id, PROTOCOL, TARGETS),
            )
            start_on_gpu(process, gpu)
            processes.append(process)
        for process in processes:
            process.join()
        failures = [process.exitcode for process in processes if process.exitcode != 0]
        if failures:
            raise RuntimeError(f"mechanism evaluation worker failures: {failures}")
        audit = audit_cells(args.output, jobs, args.episodes, PROTOCOL, TARGETS)
        audit.update(
            checkpoint_steps=list(CHECKPOINT_STEPS),
            cross_graph_probabilities=[arm.cross_graph_prob for arm in ARMS if arm.emb_dim == 256],
            embedding_dimensions=sorted({arm.emb_dim for arm in ARMS}),
        )
        atomic_json(args.output / "audit.json", audit)
        atomic_json(args.output / "status.json", {"status": "complete", "started": started,
                                                     "completed": time.time(), "invocation_id": invocation_id,
                                                     "audit": audit})
        print(json.dumps(audit, indent=2))
    except BaseException:
        import traceback
        atomic_json(args.output / "status.json", {"status": "failed", "started": started,
                                                     "completed": time.time(), "invocation_id": invocation_id,
                                                     "error": traceback.format_exc()})
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        raise


if __name__ == "__main__":
    main()
