"""Matched free/frozen readout training: three sources, three seeds, CPU only."""
import argparse
from pathlib import Path
import sys
import time

from experiments.params import get_params
from experiments.run_shared_graph import validate_configs
from .readout_training_constraint import CONDITIONS, SOURCES, configure_trainer
from .run_cpu_member_training import execute_cpu_plans

HERE = Path(__file__).resolve().parent
BASE = HERE / "member_configs/00_memberctl_ukr_rus_lowest_sorted_s0.yaml"


def make_plans(run_dir, workers=2, smoke_steps=0):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    plans = []
    for source in SOURCES:
        for seed in range(3):
            for condition in CONDITIONS:
                p = get_params(["--config", str(BASE), "--device", "123", "--seed", str(seed),
                                "--neighbor_sampling_source_subset", source,
                                "--neighbor_matching_member_seed", str(280100 + seed)])
                prefix = f"readouttrain_{source}_{condition}_s{seed}"
                p.update(prefix=prefix, exp_name=f"{'smoke_' if smoke_steps else ''}{prefix}_{stamp}_{len(plans):03d}",
                         workers=workers, loader_start_method="spawn", state_dir=str(run_dir / "state"),
                         log_dir=str(run_dir / "log"))
                if smoke_steps:
                    p.update(epochs=1, dataset_len_cap=smoke_steps, checkpoint_steps=f"0,{smoke_steps}")
                plans.append(p)
    # The intervention is model/optimizer-only, added after the strict shared-data check.
    validate_configs(plans)
    for p, condition in zip(plans, CONDITIONS * 9):
        p["readout_training_condition"] = condition
    return plans


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--models", type=int, default=6)
    parser.add_argument("--threads-per-model", type=int, default=8)
    parser.add_argument("--workers-per-model", type=int, default=2)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--evaluate", action="store_true", help="After the complete validity gate, replay both fixed test streams.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    if args.smoke_steps and not 8 <= args.smoke_steps <= 100:
        raise ValueError("smoke must be 8..100 steps")
    if args.smoke_steps and args.evaluate:
        raise ValueError("smoke results must not enter substantive target evaluation")
    plans = make_plans(args.run_dir, args.workers_per_model, args.smoke_steps)
    print(f"18 CPU models: three sources, three seeds, free/frozen readout; {args.smoke_steps or 2500} updates each.", flush=True)
    if args.dry_run:
        return
    verify = [sys.executable, "-m", "scripts.experiments.setup.target_performance_mechanisms.verify_readout_training",
              "--run-dir", str(args.run_dir), "--output", str(args.run_dir / "verified")]
    if args.smoke_steps:
        verify += ["--allow-smoke", "--expected-steps", str(args.smoke_steps)]
    execute_cpu_plans(args, plans, [BASE] * len(plans), verify, trainer_setup=configure_trainer)
    if args.evaluate:
        from .finish_readout_training import run_evaluation
        run_evaluation(args.run_dir, args.run_dir / "evaluation")


if __name__ == "__main__":
    main()
