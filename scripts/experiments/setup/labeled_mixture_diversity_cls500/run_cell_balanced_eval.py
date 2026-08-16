#!/usr/bin/env python3
"""Run remaining trajectory-eval cells through a dynamic GPU worker queue."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path[:0] = [str(REPO_ROOT), str(HERE)]

from make_plan import control_evaluation_rows, evaluation_rows  # noqa: E402


STEP_TO_CHECKPOINT = {750: 250, 1000: 500}
TARGET_COST = {
    "covid_political": 10.4,
    "election2020": 16.0,
    "facebook_page_reference": 3.4,
    "ukr_rus_suspended": 8.5,
    "twibot20": 22.6,
}


class Job(NamedTuple):
    training_steps: int
    checkpoint_step: int
    mode: str
    target: str
    model_prefix: str
    endpoint: str

    @property
    def key(self) -> tuple[int, str, str, str]:
        return self.training_steps, self.target, self.model_prefix, self.endpoint

    @property
    def label(self) -> str:
        return (
            f"step{self.training_steps}/{self.mode}/"
            f"{self.target}/{self.model_prefix}"
        )


def result_key(row: dict) -> tuple[int, str, str, str]:
    return (
        int(row.get("training_steps", 500)),
        str(row["target"]),
        str(row["model_id"]),
        str(row.get("endpoint", "heldout")),
    )


def load_completed(directories: list[Path]) -> set[tuple[int, str, str, str]]:
    completed = set()
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in directory.glob("trajectory_step*_shard*.jsonl"):
            for line_number, line in enumerate(path.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    completed.add(result_key(json.loads(line)))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"bad result row {path}:{line_number}: {exc}") from exc
    return completed


def jobs_for(training_steps: list[int], modes: list[str]) -> list[Job]:
    jobs = []
    for step in training_steps:
        checkpoint_step = STEP_TO_CHECKPOINT[step]
        for mode in modes:
            rows = control_evaluation_rows() if mode == "controls" else evaluation_rows()
            for row in rows:
                jobs.append(Job(
                    training_steps=step,
                    checkpoint_step=checkpoint_step,
                    mode=mode,
                    target=str(row["target"]),
                    model_prefix=str(row["prefix"]),
                    endpoint=str(row.get("endpoint", "heldout")),
                ))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0 1 0 1 0")
    parser.add_argument(
        "--state-root", type=Path,
        default=Path("/dataMeR1/phil/gfm/prodigy-mixconv/state_labmix500_continuation"),
    )
    parser.add_argument("--completed-dir", type=Path, action="append", default=[])
    parser.add_argument("--training-steps", type=int, nargs="+", choices=sorted(STEP_TO_CHECKPOINT), default=[750, 1000])
    parser.add_argument("--modes", nargs="+", choices=("heldout", "controls"), default=["heldout", "controls"])
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument(
        "--threads-per-process", type=int, default=0,
        help="set BLAS/OpenMP thread caps; zero preserves the current environment",
    )
    parser.add_argument("--cell-timeout", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpus = args.gpus.split()
    if not gpus or any(gpu not in {"0", "1"} for gpu in gpus):
        parser.error("--gpus must contain only owned Tucker GPUs 0 and 1")
    if args.threads_per_process < 0 or args.dataloader_workers < 0:
        parser.error("thread and DataLoader worker counts must be nonnegative")

    completed_dirs = [HERE, *args.completed_dir]
    completed = load_completed(completed_dirs)
    all_jobs = jobs_for(args.training_steps, args.modes)
    pending = [job for job in all_jobs if job.key not in completed]
    # Longest processing times first keeps expensive targets from forming a tail.
    pending.sort(key=lambda job: (-TARGET_COST[job.target], job.label))
    print(
        f"cells total={len(all_jobs)} complete={len(all_jobs) - len(pending)} "
        f"pending={len(pending)} workers={len(gpus)} gpus={gpus}",
        flush=True,
    )
    if args.dry_run:
        for job in pending:
            print(f"DRY {job.label}")
        return 0
    if not pending:
        return 0

    work: queue.Queue[Job] = queue.Queue()
    for job in pending:
        work.put(job)
    failures: list[tuple[Job, str]] = []
    failure_lock = threading.Lock()

    def worker(worker_id: int, gpu: str) -> None:
        log_path = HERE / "run_logs" / f"cell_worker{worker_id}_gpu{gpu}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_handle:
            while True:
                try:
                    job = work.get_nowait()
                except queue.Empty:
                    return
                command = [
                    sys.executable, "-u", str(HERE / "evaluate.py"),
                    "--device", gpu,
                    "--shard-index", "0", "--num-shards", "1",
                    "--mode", job.mode,
                    "--target-only", job.target,
                    "--model-prefix-only", job.model_prefix,
                    "--workers", str(args.dataloader_workers),
                    "--state-root", str(args.state_root),
                    "--checkpoint-prefix", "labmixcont",
                    "--checkpoint-step", str(job.checkpoint_step),
                    "--training-steps", str(job.training_steps),
                    "--run-stamp", f"cellbal_seed0_step{job.training_steps}",
                    "--results", str(HERE / f"trajectory_step{job.training_steps}_{job.mode}.jsonl"),
                    "--result-shard-label", f"cell{worker_id}",
                ]
                environment = os.environ.copy()
                if args.threads_per_process:
                    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                        environment[name] = str(args.threads_per_process)
                print(f"[worker {worker_id} gpu {gpu}] START {job.label}", flush=True)
                error = ""
                for attempt in range(args.retries + 1):
                    try:
                        result = subprocess.run(
                            command,
                            cwd=REPO_ROOT,
                            env=environment,
                            stdout=log_handle,
                            stderr=subprocess.STDOUT,
                            timeout=args.cell_timeout,
                            check=False,
                        )
                        log_handle.flush()
                        if result.returncode == 0:
                            error = ""
                            break
                        error = f"exit {result.returncode}"
                    except subprocess.TimeoutExpired:
                        error = f"timeout after {args.cell_timeout}s"
                    print(
                        f"[worker {worker_id} gpu {gpu}] RETRY {attempt + 1} "
                        f"{job.label}: {error}",
                        flush=True,
                    )
                if error:
                    with failure_lock:
                        failures.append((job, error))
                    print(f"[worker {worker_id} gpu {gpu}] FAILED {job.label}: {error}", flush=True)
                else:
                    print(f"[worker {worker_id} gpu {gpu}] DONE {job.label}", flush=True)
                work.task_done()

    threads = [
        threading.Thread(target=worker, args=(worker_id, gpu), daemon=False)
        for worker_id, gpu in enumerate(gpus)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if failures:
        print(f"FAILED_CELLS={len(failures)}", flush=True)
        for job, error in failures:
            print(f"FAILED {job.label}: {error}", flush=True)
        return 1
    print("ALL_PENDING_CELLS_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
