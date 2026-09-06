"""Bounded exact-workload CPU timing check on Tucker; never research results."""
import argparse
import json
import os
from pathlib import Path
import time

# Set before torch imports or spawned loader startup. Never touch occupied GPUs.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("WANDB_MODE", "offline")

import torch

from experiments.params import get_params
from experiments.run_single_experiment import load_dataset, seed_everything
from experiments.run_shared_graph import prepare_shared_dataset
from experiments.trainer import TrainerFS
from .verify_member_training import audit_consumed, model_digest


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--config", default=str(Path(__file__).parent / "member_configs/00_memberctl_ukr_rus_lowest_sorted_s0.yaml"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if not 8 <= args.steps <= 100 or not 1 <= args.threads <= 16 or not 0 <= args.workers <= 8:
        raise ValueError("invalid bounded timing budget")
    if torch.cuda.is_available() or args.output.exists():
        raise RuntimeError("GPU visible or output directory already exists")
    available = {line.split(':')[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
    if available["MemAvailable"] < 512 * 1024**2:
        raise RuntimeError("insufficient host memory headroom")
    shm = os.statvfs("/dev/shm")
    if shm.f_bavail * shm.f_frsize < 200 * 1024**3:
        raise RuntimeError("insufficient shared-memory headroom")
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.multiprocessing.set_sharing_strategy("file_system")
    params = get_params(["--config", args.config, "--device", "123", "--epochs", "1",
                        "--dataset_len_cap", str(args.steps), "--workers", str(args.workers),
                        "--loader_start_method", "spawn", "--checkpoint_steps", f"0,{args.steps}",
                        "--state_dir", str(args.output / "state"), "--log_dir", str(args.output / "log"),
                        "--prefix", "cpu_timing_only", "--timestamp", "fixed"])
    print(json.dumps({"steps": args.steps, "threads": args.threads, "workers": args.workers,
                      "graph": params["graph_filename"], "source": params["neighbor_sampling_source_subset"],
                      "research_result": False}), flush=True)
    started = time.monotonic()
    dataset = prepare_shared_dataset(load_dataset(params))
    setup_seconds = time.monotonic() - started
    seed_everything(params)
    trainer = TrainerFS(dataset, params)
    timings = []
    trainer.training_step_observer = lambda step: timings.append((step, time.monotonic()))
    train_start = time.monotonic()
    trainer.train()
    train_seconds = time.monotonic() - train_start
    audit = audit_consumed(Path(trainer.logging_dir) / "consumed_episodes.jsonl.gz", params, args.steps)
    initial = torch.load(Path(trainer.ckpt_dir) / "state_dict_0.ckpt", map_location="cpu", weights_only=True)["model"]
    terminal = torch.load(Path(trainer.ckpt_dir) / f"state_dict_{args.steps}.ckpt", map_location="cpu", weights_only=True)["model"]
    if model_digest(initial) == model_digest(terminal) or len(timings) != args.steps:
        raise RuntimeError("timing run did not complete finite weight updates")
    steady = timings[4:]
    result = {"setup_seconds": setup_seconds, "train_seconds": train_seconds,
              "steady_seconds_per_step": (steady[-1][1] - steady[0][1]) / (steady[-1][0] - steady[0][0]),
              "steps": args.steps, "threads": args.threads, "workers": args.workers,
              "consumed_audit": audit["summary"], "research_result": False, "cuda_visible": False}
    (args.output / "DONE.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
