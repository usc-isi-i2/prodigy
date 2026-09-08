#!/usr/bin/env python3
"""Evaluate optimized paper-replication runs on one fixed NM receiver panel."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
DEFAULT_EVAL_CONFIG = ROOT / "scripts/experiments/setup/final_core/training.yaml"

from scripts.experiments.setup.nm_interventions_overnight.plan import TARGETS
from scripts.experiments.setup.paper_three_seed.make_plan import build_plan


PROTOCOL = "paper_core_fixed_nm_v1"
GROUP_FAMILIES = {
    "onehop": {"ladder_1hop"},
    "twohop": {"ladder_2hop", "fixed_exposure_2hop"},
}
GROUP_COUNTS = {"onehop": 18, "twohop": 46}


def repo_relative_config(value: str | Path) -> str:
    parts = Path(value).parts
    try:
        index = parts.index("scripts")
    except ValueError as exc:
        raise ValueError(f"config is not beneath scripts/: {value}") from exc
    return "/".join(parts[index:])


def parse_run_group(value: str) -> tuple[str, Path]:
    kind, separator, path = value.partition("=")
    if not separator or kind not in GROUP_FAMILIES or not path:
        raise argparse.ArgumentTypeError("run groups must be onehop=PATH or twohop=PATH")
    return kind, Path(path).resolve()


def plan_registry(kind: str) -> dict[str, object]:
    allowed = GROUP_FAMILIES[kind]
    rows = [row for row in build_plan() if row.family in allowed]
    registry = {repo_relative_config(row.config): row for row in rows}
    if len(registry) != len(rows):
        raise ValueError(f"ambiguous {kind} config registry")
    return registry


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(handle, "sha256").hexdigest()
        return hashlib.sha256(handle.read()).hexdigest()


def normalize_sources(value: object) -> list[str]:
    if isinstance(value, str):
        return [part for part in value.split(",") if part]
    if isinstance(value, list):
        return [str(part) for part in value]
    raise ValueError(f"invalid source subset: {value!r}")


def validate_terminal_status(run_dir: Path, expected: int) -> dict:
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        raise FileNotFoundError(f"missing terminal status: {status_path}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    finished = status.get("finished", [])
    if status.get("status") != "complete":
        raise ValueError(f"non-complete run group: {run_dir}: {status.get('status')}")
    if len(finished) != expected or any(row.get("exitcode") != 0 for row in finished):
        raise ValueError(f"invalid completion ledger: {run_dir}: {len(finished)} != {expected}")
    return status


def discover(groups: list[tuple[str, Path]]) -> list[dict]:
    jobs: list[dict] = []
    for kind, run_dir in groups:
        registry = plan_registry(kind)
        validate_terminal_status(run_dir, GROUP_COUNTS[kind])
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        group_jobs = []
        for result_path in sorted(run_dir.glob("job_*/result.json")):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") != "complete":
                raise ValueError(f"non-complete job under completed group: {result_path}")
            params = json.loads((result_path.parent / "effective_config.json").read_text(encoding="utf-8"))
            config_key = repo_relative_config(params["config"])
            if config_key not in registry:
                raise ValueError(f"unexpected {kind} config: {config_key}")
            planned = registry[config_key]
            checkpoint = Path(result["checkpoint_dir"]) / f"state_dict_{planned.target_step}.ckpt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            seed = int(params["seed"])
            model_id = f"paper3seed_{planned.family}_{planned.arm}_s{seed}"
            group_jobs.append(
                {
                    "model_id": model_id,
                    "family": planned.family,
                    "arm": planned.arm,
                    "seed": seed,
                    "sources": normalize_sources(params["neighbor_sampling_source_subset"]),
                    "checkpoint": str(checkpoint),
                    "checkpoint_step": int(planned.target_step),
                    "checkpoint_sha256": sha256(checkpoint),
                    "training_revision": manifest["revision"],
                    "training_exp_name": params["exp_name"],
                    "training_config": config_key,
                    "params": params,
                }
            )
        if len(group_jobs) != GROUP_COUNTS[kind]:
            raise ValueError(f"{kind} model count mismatch: {len(group_jobs)} != {GROUP_COUNTS[kind]}")
        jobs.extend(group_jobs)
    ids = [job["model_id"] for job in jobs]
    if len(ids) != len(set(ids)):
        duplicates = sorted({value for value in ids if ids.count(value) > 1})
        raise ValueError(f"duplicate core model ids: {duplicates[:10]}")
    return jobs


def completed_cell(path: Path, job: dict, target: str, episodes: int,
                   protocol: str | None = None) -> bool:
    protocol = PROTOCOL if protocol is None else protocol
    if not path.is_file():
        return False
    old = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol": protocol,
        "model_id": job["model_id"],
        "family": job["family"],
        "arm": job["arm"],
        "seed": job["seed"],
        "target": target,
        "episodes": episodes,
        "sources": job["sources"],
        "checkpoint": job["checkpoint"],
        "checkpoint_step": job["checkpoint_step"],
        "checkpoint_sha256": job["checkpoint_sha256"],
        "training_revision": job["training_revision"],
    }
    drift = {key: (old.get(key), value) for key, value in expected.items() if old.get(key) != value}
    if drift:
        raise ValueError(f"resume metadata mismatch: {path}: {drift}")
    if not all(math.isfinite(float(old[key])) for key in ("roc_auc", "accuracy", "loss")):
        raise ValueError(f"non-finite completed cell: {path}")
    return True


def worker(dataset, jobs: list[dict], eval_params: dict, output: str, worker_id: int,
           episodes: int, invocation_id: str, protocol: str | None = None,
           targets: tuple[str, ...] | None = None) -> None:
    protocol = PROTOCOL if protocol is None else protocol
    targets = TARGETS if targets is None else targets
    os.setsid()
    import torch
    from experiments.nm_campaign import atomic_json, evaluate, materialize
    from scripts.experiments.setup.nm_interventions_overnight.evaluate import model_from_params

    output_path = Path(output)
    stream = (output_path / f"worker_{worker_id}.log").open("a", buffering=1)
    os.dup2(stream.fileno(), 1)
    os.dup2(stream.fileno(), 2)
    torch.set_num_threads(2)
    torch.autograd.set_detect_anomaly(False)
    device = torch.device("cuda:0")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    record = {"invocation_id": invocation_id, "pid": os.getpid(), "started": time.time()}
    atomic_json(output_path / f"worker_{worker_id}_status.json", {"status": "running", **record})
    try:
        for target in targets:
            pending = [
                job for job in jobs
                if not completed_cell(output_path / "cells" / job["model_id"] / f"{target}.json",
                                      job, target, episodes, protocol)
            ]
            if not pending:
                continue
            batches, fingerprint = materialize(dataset, eval_params, target, "test", episodes)
            for job in pending:
                model = model_from_params(job["params"], job["checkpoint"], device)
                metrics = evaluate(model, batches, device)
                payload = {
                    "protocol": protocol,
                    "model_id": job["model_id"],
                    "family": job["family"],
                    "arm": job["arm"],
                    "seed": job["seed"],
                    "target": target,
                    "sources": job["sources"],
                    "checkpoint": job["checkpoint"],
                    "checkpoint_step": job["checkpoint_step"],
                    "checkpoint_sha256": job["checkpoint_sha256"],
                    "training_revision": job["training_revision"],
                    "training_exp_name": job["training_exp_name"],
                    "training_config": job["training_config"],
                    "evaluation_revision": revision,
                    "fingerprint": fingerprint,
                    "invocation_id": invocation_id,
                    **metrics,
                }
                atomic_json(output_path / "cells" / job["model_id"] / f"{target}.json", payload)
                print(f"DONE {job['model_id']} {target} auc={metrics['roc_auc']:.6f}", flush=True)
                del model
                gc.collect()
            del batches
            gc.collect()
        atomic_json(output_path / f"worker_{worker_id}_status.json",
                    {"status": "complete", "completed": time.time(), **record})
    except BaseException:
        atomic_json(output_path / f"worker_{worker_id}_status.json",
                    {"status": "failed", "error": traceback.format_exc(), **record})
        traceback.print_exc()
        raise


def audit_cells(output: Path, jobs: list[dict], episodes: int,
                protocol: str | None = None,
                targets: tuple[str, ...] | None = None) -> dict:
    protocol = PROTOCOL if protocol is None else protocol
    targets = TARGETS if targets is None else targets
    expected = {(job["model_id"], target) for job in jobs for target in targets}
    rows = []
    for path in sorted((output / "cells").glob("*/*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        rows.append(row)
    keys = [(row.get("model_id"), row.get("target")) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate core evaluation cells")
    observed = set(keys)
    if observed != expected:
        raise ValueError(f"core evaluation coverage mismatch: missing={len(expected-observed)} extra={len(observed-expected)}")
    for row in rows:
        if row.get("protocol") != protocol or row.get("episodes") != episodes:
            raise ValueError("core evaluation protocol drift")
        if not all(math.isfinite(float(row[key])) for key in ("roc_auc", "accuracy", "loss")):
            raise ValueError("non-finite core evaluation cell")
    fingerprints = {}
    for target in targets:
        values = {row["fingerprint"] for row in rows if row["target"] == target}
        if len(values) != 1:
            raise ValueError(f"episode fingerprint drift for {target}: {len(values)}")
        fingerprints[target] = next(iter(values))
    return {
        "status": "complete",
        "protocol": protocol,
        "models": len(jobs),
        "targets": len(targets),
        "cells": len(rows),
        "episodes_per_cell": episodes,
        "training_seeds": sorted({job["seed"] for job in jobs}),
        "families": {family: sum(job["family"] == family for job in jobs)
                     for family in sorted({job["family"] for job in jobs})},
        "fingerprints": fingerprints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-group", action="append", type=parse_run_group, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CONFIG)
    parser.add_argument("--gpus", nargs="+", type=int, choices=range(4), default=[0, 1, 2, 3])
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--expected-models", type=int, default=82)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers_per_gpu < 1 or args.episodes < 1:
        parser.error("worker and episode counts must be positive")
    jobs = discover(args.run_group)
    if len(jobs) != args.expected_models:
        raise ValueError(f"core model count mismatch: {len(jobs)} != {args.expected_models}")
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
    atomic_json(args.output / "plan.json", {
        "invocation_id": invocation_id,
        "models": [job["model_id"] for job in jobs],
        "targets": list(TARGETS),
        "episodes": args.episodes,
        "gpus": args.gpus,
        "workers_per_gpu": args.workers_per_gpu,
        "eval_config": str(args.eval_config.resolve()),
        "eval_graph_filename": eval_params["graph_filename"],
        "evaluation_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    })
    started = time.time()
    atomic_json(args.output / "status.json", {"status": "running", "started": started,
                                                "invocation_id": invocation_id})
    torch.set_num_threads(4)
    torch.autograd.set_detect_anomaly(False)
    dataset = prepare_shared_dataset(load_dataset(eval_params))
    missing_targets = sorted(set(TARGETS) - set(dataset.graph.source_graph_names))
    if missing_targets:
        raise ValueError(f"evaluation graph lacks registered targets: {missing_targets}")
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
            raise RuntimeError(f"core evaluation worker failures: {failures}")
        audit = audit_cells(args.output, jobs, args.episodes, PROTOCOL, TARGETS)
        atomic_json(args.output / "audit.json", audit)
        atomic_json(args.output / "status.json", {"status": "complete", "started": started,
                                                    "completed": time.time(), "invocation_id": invocation_id,
                                                    "audit": audit})
        print(json.dumps(audit, indent=2))
    except BaseException:
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
