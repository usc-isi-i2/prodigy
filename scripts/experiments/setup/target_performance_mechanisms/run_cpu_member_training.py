"""GPU-isolated matched-policy training using the existing shared CPU graph.

Training, sampling, and checkpoint semantics are unchanged. Only orchestration
and the common training device differ from the GPU launcher. Validate concurrent
smoke runs before using this path for substantive mechanism evidence.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("WANDB_MODE", "offline")

import torch

from experiments.params import get_params
from experiments.run_shared_graph import prepare_shared_dataset, shared_storage_report, validate_configs, write_json
from experiments.run_single_experiment import load_dataset, seed_everything
from experiments.trainer import TrainerFS


def make_plan(configs, run_dir, workers, smoke_steps):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    plans = []
    for index, config in enumerate(configs):
        params = get_params(["--config", str(config), "--device", "123"])
        params.update(workers=workers, loader_start_method="spawn", state_dir=str(run_dir / "state"),
                      log_dir=str(run_dir / "log"), exp_name=f"{'smoke_' if smoke_steps else ''}{params['prefix']}_cpu_{stamp}_{index:03d}")
        if smoke_steps:
            params.update(epochs=1, dataset_len_cap=smoke_steps, checkpoint_steps=f"0,{smoke_steps}")
        plans.append(params)
    validate_configs(plans)
    if len(plans) != 24 or any(p["device"].type != "cpu" or not p["train_episode_audit"] for p in plans):
        raise ValueError("requires all 24 audited CPU arms")
    return plans


def train_cpu(dataset, params, job_dir, threads, trainer_setup=None):
    os.setsid()
    job_dir = Path(job_dir)
    with (job_dir / "console.log").open("a", buffering=1) as stream:
        os.dup2(stream.fileno(), 1)
        os.dup2(stream.fileno(), 2)
        result = {"status": "initializing", "pid": os.getpid(), "device": "cpu", "started": time.time(),
                  "shared_storage": shared_storage_report(dataset)}
        write_json(job_dir / "result.json", result)
        try:
            if torch.cuda.is_available() or not all(result["shared_storage"].values()):
                raise RuntimeError("GPU visible or shared storage lost in child")
            torch.set_num_threads(threads)
            torch.multiprocessing.set_sharing_strategy("file_system")
            seed_everything(params)
            trainer = TrainerFS(dataset, params)
            if trainer_setup is not None:
                trainer_setup(trainer)
            observed = []
            prior_observer = getattr(trainer, "training_step_observer", None)
            def observe(step):
                if prior_observer is not None:
                    prior_observer(step)
                observed.append((step, time.monotonic()))
            trainer.training_step_observer = observe
            result.update(status="training", training_started=time.time())
            write_json(job_dir / "result.json", result)
            trainer.train()
            steady = observed[4:]
            result.update(status="complete", completed=time.time(), checkpoint_dir=trainer.ckpt_dir,
                          steps_observed=len(observed),
                          steady_seconds_per_step=((steady[-1][1] - steady[0][1]) / (steady[-1][0] - steady[0][0])
                                                   if len(steady) > 1 else None))
            if hasattr(trainer, "training_constraint_receipt"):
                result["training_constraint"] = trainer.training_constraint_receipt
            write_json(job_dir / "result.json", result)
        except BaseException:
            result.update(status="failed", error=traceback.format_exc(), completed=time.time())
            write_json(job_dir / "result.json", result)
            traceback.print_exc()
            raise


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--configs", type=Path, default=Path(__file__).parent / "member_configs")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--models", type=int, default=6)
    p.add_argument("--threads-per-model", type=int, default=8)
    p.add_argument("--workers-per-model", type=int, default=2)
    p.add_argument("--smoke-steps", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.run_dir = args.run_dir.resolve()
    budget = args.models * (args.threads_per_model + args.workers_per_model)
    if not 1 <= args.models <= 8 or not 1 <= args.threads_per_model <= 16 or not 0 <= args.workers_per_model <= 4:
        raise ValueError("invalid bounded CPU concurrency")
    if budget > 96 or (args.smoke_steps and not 8 <= args.smoke_steps <= 100):
        raise ValueError("CPU or smoke budget exceeds bound")
    if args.run_dir.exists() or torch.cuda.is_available():
        raise RuntimeError("output exists or GPU visible")
    configs = sorted(args.configs.resolve().glob("*.yaml"))
    plans = make_plan(configs, args.run_dir, args.workers_per_model, args.smoke_steps)
    print(f"CPU-only: {len(plans)} models, {args.models} active, {budget} total tensor/loader threads; "
          f"{args.smoke_steps or 2500} steps/model; {'smoke' if args.smoke_steps else 'substantive'}", flush=True)
    if args.dry_run:
        return
    verify = [sys.executable, "-m", "scripts.experiments.setup.target_performance_mechanisms.verify_member_training",
              "--run-dir", str(args.run_dir), "--output", str(args.run_dir / "verified")]
    if args.smoke_steps:
        verify += ["--allow-smoke", "--expected-steps", str(args.smoke_steps)]
    execute_cpu_plans(args, plans, configs, verify)


def execute_cpu_plans(args, plans, configs, verify, trainer_setup=None):
    """Common bounded supervisor; callers must validate their experiment grid."""
    budget = args.models * (args.threads_per_model + args.workers_per_model)
    if (not 1 <= args.models <= 8 or not 1 <= args.threads_per_model <= 16
            or not 0 <= args.workers_per_model <= 4 or budget > 96):
        raise ValueError("invalid bounded CPU resources")
    if args.run_dir.exists() or torch.cuda.is_available():
        raise RuntimeError("output exists or GPU visible")
    if budget > len(os.sched_getaffinity(0)) // 4:
        raise RuntimeError("would exceed one quarter of available logical CPU slots")
    mem = {line.split(':')[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
    shm = os.statvfs("/dev/shm")
    if mem["MemAvailable"] < 512 * 1024**2 or shm.f_bavail * shm.f_frsize < 200 * 1024**3:
        raise RuntimeError("insufficient host/shared-memory headroom")
    args.run_dir.mkdir(parents=True)
    manifest = {"jobs": plans, "configs": list(map(str, configs)), "device": "cpu", "cpu_thread_budget": budget,
                "mode": "smoke" if args.smoke_steps else "training", "models_active": args.models,
                "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    write_json(args.run_dir / "manifest.json", manifest)
    torch.set_num_threads(4)
    torch.multiprocessing.set_sharing_strategy("file_system")
    started = time.monotonic()
    dataset = prepare_shared_dataset(load_dataset(plans[0]))
    manifest.update(shared_graph_setup_seconds=time.monotonic() - started, shared_storage=shared_storage_report(dataset),
                    source_names=list(dataset.graph.source_graph_names))
    write_json(args.run_dir / "manifest.json", manifest)
    context = torch.multiprocessing.get_context("spawn")
    active, finished, index = {}, [], 0
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        while index < len(plans) or active:
            while index < len(plans) and len(active) < args.models:
                job = args.run_dir / f"job_{index:03d}"
                job.mkdir()
                write_json(job / "effective_config.json", plans[index])
                process = context.Process(target=train_cpu, args=(dataset, plans[index], str(job), args.threads_per_model, trainer_setup))
                process.start()
                active[index] = process
                print(f"Started CPU job {index}: {plans[index]['prefix']}", flush=True)
                index += 1
            for job_index, process in list(active.items()):
                if not process.is_alive():
                    process.join()
                    finished.append({"job": job_index, "exitcode": process.exitcode})
                    del active[job_index]
                    if process.exitcode:
                        raise RuntimeError(f"CPU job {job_index} failed: {process.exitcode}")
                    print(f"Completed CPU job {job_index}", flush=True)
            time.sleep(1)
        write_json(args.run_dir / "status.json", {"status": "complete", "finished": finished})
        subprocess.run(verify, check=True)
        print("All CPU runs and consumed-stream validity checks passed.", flush=True)
    except BaseException:
        write_json(args.run_dir / "status.json", {"status": "failed", "finished": finished, "error": traceback.format_exc()})
        for process in active.values():
            if process.is_alive():
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    process.terminate()
        for process in active.values():
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
        raise


if __name__ == "__main__":
    main()
