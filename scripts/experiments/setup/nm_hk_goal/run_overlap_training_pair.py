"""Run one exact-input HK baseline/treatment pair sequentially on one GPU."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from experiments.params import get_params
from experiments.run_single_experiment import load_dataset, seed_everything
from experiments.trainer import TrainerFS


HERE = Path(__file__).resolve().parent
CONFIGS = {
    "baseline": HERE / "configs/hk_overlap_baseline.yaml",
    "treatment": HERE / "configs/hk_overlap_treatment.yaml",
}


def parameters(name, run_dir, device, steps):
    params = get_params(["--config", str(CONFIGS[name]), "--device", str(device)])
    stamp = run_dir.name
    params.update(
        exp_name=f"nm_hk_overlap_{name}_{stamp}",
        prefix=f"nm_hk_overlap_{name}",
        state_dir=str(run_dir / name / "state"),
        log_dir=str(run_dir / name / "log"),
    )
    if steps:
        params.update(
            epochs=1,
            dataset_len_cap=steps,
            val_len_cap=min(16, steps),
            test_len_cap=min(16, steps),
            checkpoint_steps=f"0,{steps}",
        )
    return params


def serializable(params):
    return {key: str(value) if isinstance(value, (Path, torch.device)) else value for key, value in params.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    if args.run_dir.exists():
        raise ValueError("choose a new run directory")
    if args.device not in range(4):
        raise ValueError("only owned Tucker GPUs 0-3 may be used")
    if args.smoke_steps and not 2 <= args.smoke_steps <= 20:
        raise ValueError("smoke must use 2..20 updates")
    plans = {name: parameters(name, args.run_dir, args.device, args.smoke_steps) for name in CONFIGS}
    controlled = {
        key for key in plans["baseline"]
        if plans["baseline"].get(key) != plans["treatment"].get(key)
    }
    expected = {
        "config", "exp_name", "prefix", "log_dir", "state_dir",
        "neighbor_matching_overlap_aware_support_edges",
    }
    if not controlled <= expected:
        raise ValueError(f"uncontrolled pair parameters: {sorted(controlled - expected)}")
    print(json.dumps({
        "conditions": list(CONFIGS), "steps_each": args.smoke_steps or 2500,
        "sequential": True, "device": args.device, "workers": 0,
        "expected_minutes_each": "15-20 full; under 2 smoke",
        "full_graph_loads": 1,
    }, indent=2), flush=True)
    if args.dry_run:
        return

    args.run_dir.mkdir(parents=True)
    started = time.time()
    dataset_params = plans["baseline"].copy()
    seed_everything(dataset_params)
    dataset = load_dataset(dataset_params)
    completed = {}
    for name, params in plans.items():
        seed_everything(params)
        trainer = TrainerFS(dataset, params)
        (Path(trainer.logging_dir) / "effective_config.json").write_text(
            json.dumps(serializable(trainer.parameter), indent=2) + "\n"
        )
        trainer.train()
        completed[name] = {
            "seconds": time.time() - started - sum(value["seconds"] for value in completed.values()),
            "logging_dir": trainer.logging_dir,
            "state_dir": trainer.state_dir,
        }
    verify = [
        sys.executable, "-m",
        "scripts.experiments.setup.nm_hk_goal.verify_overlap_training_pair",
        "--baseline", str(args.run_dir / "baseline"),
        "--treatment", str(args.run_dir / "treatment"),
        "--out", str(args.run_dir / "verification.json"),
        "--expected-steps", str(args.smoke_steps or 2500),
    ]
    subprocess.run(verify, check=True)
    (args.run_dir / "DONE.json").write_text(json.dumps({
        "complete": True,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "conditions": completed,
        "verification": str(args.run_dir / "verification.json"),
        "smoke": bool(args.smoke_steps),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
