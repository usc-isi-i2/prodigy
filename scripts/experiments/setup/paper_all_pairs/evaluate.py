#!/usr/bin/env python3
"""Evaluate the complete three-seed source-pair panel on all nine NM targets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[4]
from scripts.experiments.setup.nm_interventions_overnight.plan import TARGETS
from scripts.experiments.setup.paper_all_pairs.plan import SEEDS, SOURCES, STEPS, build_pairs
from scripts.experiments.setup.paper_three_seed.evaluate_fast_core import (
    audit_cells,
    normalize_sources,
    sha256,
    validate_terminal_status,
    worker,
)


PROTOCOL = "paper_all_pairs_fixed_nm_v1"
EXPECTED_MODELS = len(build_pairs()) * len(SEEDS)


def discover(run_dir: Path) -> list[dict]:
    validate_terminal_status(run_dir, EXPECTED_MODELS)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    allowed = {pair.sources: pair for pair in build_pairs()}
    order = {source: index for index, source in enumerate(SOURCES)}
    jobs = []
    observed = set()
    for result_path in sorted(run_dir.glob("job_*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            raise ValueError(f"non-complete job under completed run: {result_path}")
        params = json.loads((result_path.parent / "effective_config.json").read_text(encoding="utf-8"))
        sources = tuple(sorted(normalize_sources(params["neighbor_sampling_source_subset"]), key=order.__getitem__))
        if sources not in allowed:
            raise ValueError(f"unexpected source pair in {result_path}: {sources}")
        pair = allowed[sources]
        seed = int(params["seed"])
        if seed not in SEEDS:
            raise ValueError(f"unexpected seed in {result_path}: {seed}")
        key = (pair.arm, seed)
        if key in observed:
            raise ValueError(f"duplicate pair/seed job: {key}")
        observed.add(key)
        checkpoint = Path(result["checkpoint_dir"]) / f"state_dict_{STEPS}.ckpt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        jobs.append({
            "model_id": f"paper_pair_{pair.arm}_s{seed}",
            "family": "all_pairs",
            "arm": pair.arm,
            "seed": seed,
            "sources": list(sources),
            "checkpoint": str(checkpoint),
            "checkpoint_step": STEPS,
            "checkpoint_sha256": sha256(checkpoint),
            "training_revision": manifest["revision"],
            "training_exp_name": params["exp_name"],
            "training_config": str(params["config"]),
            "params": params,
        })
    expected = {(pair.arm, seed) for pair in build_pairs() for seed in SEEDS}
    if observed != expected:
        raise ValueError(f"pair coverage mismatch: missing={len(expected-observed)} extra={len(observed-expected)}")
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path,
                        default=ROOT / "scripts/experiments/setup/final_core/training.yaml")
    parser.add_argument("--gpus", nargs="+", type=int, choices=range(4), default=[0, 1, 2, 3])
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers_per_gpu < 1 or args.episodes < 1:
        parser.error("worker and episode counts must be positive")
    jobs = discover(args.run_dir.resolve())
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
    atomic_json(args.output / "plan.json", {
        "invocation_id": invocation_id,
        "models": [job["model_id"] for job in jobs],
        "targets": list(TARGETS),
        "episodes": args.episodes,
        "gpus": args.gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "evaluation_revision": revision,
    })
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
            raise RuntimeError(f"pair evaluation worker failures: {failures}")
        audit = audit_cells(args.output, jobs, args.episodes, PROTOCOL, TARGETS)
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
