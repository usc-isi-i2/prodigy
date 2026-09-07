"""Resource guard and explicit command for the existing shared-graph trainer."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--configs", type=Path, default=Path(__file__).parent / "member_configs")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--gpus", type=int, nargs="+", required=True, choices=(0, 1, 2, 3))
    p.add_argument("--models-per-gpu", type=int, default=6)
    p.add_argument("--workers-per-model", type=int, default=4)
    p.add_argument("--threads-per-model", type=int, default=2)
    p.add_argument("--smoke-steps", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    configs = sorted(args.configs.glob("*.yaml"))
    if not configs or args.run_dir.exists():
        raise ValueError("configs missing or run directory already exists")
    worker_budget = min(len(configs), len(args.gpus) * args.models_per_gpu) * args.workers_per_model
    cmd = [sys.executable, "experiments/run_shared_graph.py", "--configs", *map(str, configs),
           "--gpus", *map(str, args.gpus), "--models-per-gpu", str(args.models_per_gpu),
           "--workers-per-model", str(args.workers_per_model), "--worker-budget", str(worker_budget),
           "--threads-per-model", str(args.threads_per_model), "--run-dir", str(args.run_dir)]
    if args.smoke_steps:
        cmd += ["--smoke-steps", str(args.smoke_steps)]
    if args.dry_run:
        cmd += ["--dry-run"]
    else:
        gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True)
        selected = set()
        for line in gpu_info.splitlines():
            index, uuid, memory = map(str.strip, line.split(","))
            if int(index) in args.gpus:
                selected.add(uuid)
                if int(memory) > 1000:
                    raise RuntimeError(f"GPU {index} is occupied; no launch")
        if len(selected) != len(set(args.gpus)):
            raise RuntimeError("requested GPU inventory unresolved")
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True)
        if selected & set(active.splitlines()):
            raise RuntimeError("selected GPU has an active compute process")
        mem = {line.split(':')[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
        if mem["MemAvailable"] < 512 * 1024**2:
            raise RuntimeError("need at least 512 GiB available host memory")
        shm = os.statvfs("/dev/shm")
        if shm.f_bavail * shm.f_frsize < 200 * 1024**3:
            raise RuntimeError("need at least 200 GiB available shared memory")
    print(shlex.join(cmd), flush=True)
    os.environ.setdefault("WANDB_MODE", "offline")
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
