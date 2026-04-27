#!/usr/bin/env python3
"""
Generate cross-dataset evaluation jobs for all model/dataset/task combinations.
Creates a SLURM batch script with all evaluation commands.
"""

import sys
import os
from pathlib import Path
from datetime import datetime


# Models to evaluate - from your provided list
# Base directory: /home1/eibl/gfm/prodigy/log or wherever your experiments are stored
EXPERIMENT_BASE = "/home1/eibl/gfm/prodigy/log"

MODELS = [
    f"{EXPERIMENT_BASE}/train2_midterm_nm_to_covid_nm_16_04_2026_10_07_00/state_dict",
    f"{EXPERIMENT_BASE}/exp2_train2_midterm_nm_to_ukr_rus_nm_16_04_2026_10_32_38/checkpoint/state_dict_27000.ckpt",
    f"{EXPERIMENT_BASE}/exp3_train2_covid_nm_to_ukr_rus_nm_16_04_2026_13_39_26/checkpoint/state_dict_30000.ckpt",
    f"{EXPERIMENT_BASE}/exp4_train2_midterm_lp_to_covid_lp_16_04_2026_17_31_13/state_dict",
    f"{EXPERIMENT_BASE}/exp5_train2_midterm_lp_to_ukr_rus_lp_16_04_2026_17_48_24/state_dict",
    f"{EXPERIMENT_BASE}/exp6_train2_covid_lp_to_ukr_rus_lp_16_04_2026_17_24_43/state_dict",
    f"{EXPERIMENT_BASE}/exp7_train2_midterm_nm_to_covid_lp_17_04_2026_16_35_55/state_dict",
    f"{EXPERIMENT_BASE}/exp8_train2_midterm_nm_to_ukr_rus_lp_17_04_2026_16_36_31/state_dict",
    f"{EXPERIMENT_BASE}/exp9_train2_covid_nm_to_ukr_rus_lp_17_04_2026_16_36_23/state_dict",
    f"{EXPERIMENT_BASE}/exp10_train2_midterm_lp_to_covid_nm_22_04_2026_13_24_06/state_dict",
    f"{EXPERIMENT_BASE}/exp11_train2_midterm_lp_to_ukr_rus_nm_22_04_2026_14_22_43/state_dict",
    f"{EXPERIMENT_BASE}/exp12_train2_covid_lp_to_ukr_rus_nm_22_04_2026_14_22_38/state_dict",
    f"{EXPERIMENT_BASE}/exp13_train2_covid_nm_to_midterm_nm_23_04_2026_13_24_00/checkpoint/state_dict_15000.ckpt",
    f"{EXPERIMENT_BASE}/exp14_train2_ukr_rus_nm_to_midterm_nm_23_04_2026_13_26_26/checkpoint/state_dict_12000.ckpt",
    f"{EXPERIMENT_BASE}/exp15_train2_ukr_rus_nm_to_covid_nm_23_04_2026_13_33_55/state_dict",
]

# Datasets to evaluate on
DATASETS = [
    "midterm",
    "covid19_twitter",
    "ukr_rus_twitter",
]

# Tasks to evaluate - adjust based on what your models were trained on
TASKS = [
    "node_masking",
    "link_prediction",
]


def generate_eval_jobs(output_file="eval_jobs.txt"):
    """Generate evaluation job list."""
    
    jobs = []
    
    for model_path in MODELS:
        for dataset in DATASETS:
            for task in TASKS:
                job_cmd = (
                    f"python /home1/eibl/gfm/prodigy/scripts/eval_cross_dataset.py "
                    f"--model_path {model_path} "
                    f"--dataset {dataset} "
                    f"--task {task} "
                    f"--device $((SLURM_ARRAY_TASK_ID % 4)) "  # Distribute GPU IDs
                    f"--output_dir /home1/eibl/gfm/prodigy/eval_results/{dataset}/{task}"
                )
                jobs.append(job_cmd)
    
    # Save to file
    with open(output_file, 'w') as f:
        for i, job in enumerate(jobs, 1):
            f.write(f"# Job {i}/{len(jobs)}\n")
            f.write(job + "\n\n")
    
    print(f"Generated {len(jobs)} evaluation jobs")
    print(f"Saved to {output_file}")
    
    return jobs


def generate_sbatch_script(num_jobs, output_file="eval_cross_dataset.sbatch"):
    """Generate SLURM batch submission script."""
    
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=cross_eval
#SBATCH --array=0-{num_jobs-1}%40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/home1/eibl/gfm/prodigy/logs/eval_%a.log
#SBATCH --error=/home1/eibl/gfm/prodigy/logs/eval_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=${{USER}}@example.com

# Create log directory
mkdir -p /home1/eibl/gfm/prodigy/logs

# Load environment
module load cuda/11.8

# Activate conda
source /home/${{USER}}/.bashrc
conda activate prodigy

# Set working directory
cd /home1/eibl/gfm/prodigy

# Run job from the list
JOB_COMMANDS=(
"""
    
    # Read job commands from file
    with open("eval_jobs.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                sbatch_content += f'  "{line}"\n'
    
    sbatch_content += f"""
)

# Get the command for this array task
COMMAND="${{JOB_COMMANDS[${{SLURM_ARRAY_TASK_ID}}]}}"

echo "Running task ${{SLURM_ARRAY_TASK_ID}}: $COMMAND"
eval "$COMMAND"
EXIT_CODE=$?

echo "Task ${{SLURM_ARRAY_TASK_ID}} completed with exit code $EXIT_CODE"
exit $EXIT_CODE
"""
    
    with open(output_file, 'w') as f:
        f.write(sbatch_content)
    
    print(f"Generated SLURM script: {output_file}")
    return sbatch_content


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate cross-dataset evaluation jobs")
    parser.add_argument("--job_list", type=str, default="eval_jobs.txt",
                       help="Output file for job list")
    parser.add_argument("--sbatch_script", type=str, default="eval_cross_dataset.sbatch",
                       help="Output SLURM batch script")
    
    args = parser.parse_args()
    
    # Generate jobs
    jobs = generate_eval_jobs(args.job_list)
    
    # Generate SLURM script
    generate_sbatch_script(len(jobs), args.sbatch_script)
    
    print("\n" + "="*80)
    print(f"Total evaluation jobs: {len(jobs)}")
    print(f"Models: {len(MODELS)}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Tasks: {len(TASKS)}")
    print(f"Total combinations: {len(MODELS) * len(DATASETS) * len(TASKS)}")
    print("="*80)
    
    print(f"\nNext steps:")
    print(f"1. Review {args.job_list} to verify job commands")
    print(f"2. Submit jobs: sbatch {args.sbatch_script}")
    print(f"3. Monitor: squeue -u $USER")
