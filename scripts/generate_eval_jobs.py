#!/usr/bin/env python3
"""Generate cross-dataset evaluation jobs and a matching SLURM array script."""

import argparse


# Models to evaluate - from your provided list
# Base directory: /home1/eibl/gfm/prodigy/log or wherever your experiments are stored
EXPERIMENT_BASE = "/home1/eibl/gfm/prodigy/log"
DEFAULT_PROJECT_ROOT = "/home1/eibl/gfm/prodigy"
DEFAULT_OUTPUT_ROOT = f"{DEFAULT_PROJECT_ROOT}/eval_results"

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


def generate_eval_jobs(
    output_file="eval_jobs.txt",
    project_root=DEFAULT_PROJECT_ROOT,
    output_root=DEFAULT_OUTPUT_ROOT,
):
    """Generate the evaluation command list."""
    jobs = []

    for model_path in MODELS:
        for dataset in DATASETS:
            for task in TASKS:
                job_cmd = (
                    f"python {project_root}/scripts/eval_cross_dataset.py "
                    f"--model_path {model_path} "
                    f"--dataset {dataset} "
                    f"--task {task} "
                    f"--device 0 "
                    f"--output_dir {output_root}/{dataset}/{task}"
                )
                jobs.append(job_cmd)

    with open(output_file, "w") as f:
        for i, job in enumerate(jobs, 1):
            f.write(f"# Job {i}/{len(jobs)}\n")
            f.write(job + "\n\n")

    print(f"Generated {len(jobs)} evaluation jobs")
    print(f"Saved to {output_file}")
    return jobs


def generate_sbatch_script(
    num_jobs,
    output_file="eval_cross_dataset.sbatch",
    job_list_file="eval_jobs.txt",
    project_root=DEFAULT_PROJECT_ROOT,
    partition="gpu",
    array_parallel=16,
    cpus_per_task=8,
    mem="32G",
    time_limit="4:00:00",
    gpus_per_task=1,
    gpu_type=None,
    conda_env="prodigy",
    cuda_module=None,
    mail_type=None,
    mail_user=None,
):
    """Generate the SLURM array submission script."""
    logs_dir = f"{project_root}/logs"
    gres_value = f"gpu:{gpus_per_task}"
    if gpu_type:
        gres_value = f"gpu:{gpu_type}:{gpus_per_task}"

    sbatch_lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=cross_eval",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --array=0-{num_jobs - 1}%{array_parallel}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --gres={gres_value}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --output={logs_dir}/eval_%A_%a.out",
        f"#SBATCH --error={logs_dir}/eval_%A_%a.err",
    ]

    if mail_type and mail_user:
        sbatch_lines.append(f"#SBATCH --mail-type={mail_type}")
        sbatch_lines.append(f"#SBATCH --mail-user={mail_user}")

    sbatch_lines.extend(
        [
            "",
            "set -euo pipefail",
            "",
            f"mkdir -p {logs_dir}",
        ]
    )

    if cuda_module:
        sbatch_lines.append(f"module load {cuda_module}")

    sbatch_lines.extend(
        [
            'source "$(conda info --base)/etc/profile.d/conda.sh"',
            f"conda activate {conda_env}",
            "",
            f"cd {project_root}",
            "",
            "JOB_COMMANDS=(",
        ]
    )

    with open(job_list_file, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line and not line.startswith("#"):
                sbatch_lines.append(f'  "{line}"')

    sbatch_lines.extend(
        [
            ")",
            "",
            'COMMAND="${JOB_COMMANDS[${SLURM_ARRAY_TASK_ID}]}"',
            "",
            'echo "Running task ${SLURM_ARRAY_TASK_ID}: $COMMAND"',
            'echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"',
            'eval "$COMMAND"',
            "EXIT_CODE=$?",
            "",
            'echo "Task ${SLURM_ARRAY_TASK_ID} completed with exit code $EXIT_CODE"',
            "exit $EXIT_CODE",
            "",
        ]
    )

    sbatch_content = "\n".join(sbatch_lines)
    with open(output_file, "w") as f:
        f.write(sbatch_content)

    print(f"Generated SLURM script: {output_file}")
    return sbatch_content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cross-dataset evaluation jobs")
    parser.add_argument("--job_list", type=str, default="eval_jobs.txt",
                       help="Output file for job list")
    parser.add_argument("--sbatch_script", type=str, default="eval_cross_dataset.sbatch",
                       help="Output SLURM batch script")
    parser.add_argument("--project_root", type=str, default=DEFAULT_PROJECT_ROOT,
                       help="Cluster path to the prodigy checkout")
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT,
                       help="Root directory for evaluation outputs")
    parser.add_argument("--partition", type=str, default="gpu",
                       help="SLURM partition name")
    parser.add_argument("--array_parallel", type=int, default=16,
                       help="Maximum number of array tasks to run concurrently")
    parser.add_argument("--cpus_per_task", type=int, default=8,
                       help="CPU cores per evaluation task")
    parser.add_argument("--mem", type=str, default="32G",
                       help="Memory request per evaluation task")
    parser.add_argument("--time_limit", type=str, default="4:00:00",
                       help="Wall clock limit per evaluation task")
    parser.add_argument("--gpus_per_task", type=int, default=1,
                       help="GPUs requested per evaluation task")
    parser.add_argument("--gpu_type", type=str, default=None,
                       help="Optional GPU type, e.g. a100, a40, v100, p100, l40s")
    parser.add_argument("--conda_env", type=str, default="prodigy",
                       help="Conda environment to activate")
    parser.add_argument("--cuda_module", type=str, default=None,
                       help="Optional CUDA module to load before activation")
    parser.add_argument("--mail_type", type=str, default=None,
                       help="Optional SLURM mail type, e.g. FAIL")
    parser.add_argument("--mail_user", type=str, default=None,
                       help="Optional SLURM mail recipient")

    args = parser.parse_args()

    jobs = generate_eval_jobs(
        args.job_list,
        project_root=args.project_root,
        output_root=args.output_root,
    )
    generate_sbatch_script(
        len(jobs),
        args.sbatch_script,
        job_list_file=args.job_list,
        project_root=args.project_root,
        partition=args.partition,
        array_parallel=args.array_parallel,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        time_limit=args.time_limit,
        gpus_per_task=args.gpus_per_task,
        gpu_type=args.gpu_type,
        conda_env=args.conda_env,
        cuda_module=args.cuda_module,
        mail_type=args.mail_type,
        mail_user=args.mail_user,
    )

    print("\n" + "=" * 80)
    print(f"Total evaluation jobs: {len(jobs)}")
    print(f"Models: {len(MODELS)}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Tasks: {len(TASKS)}")
    print(f"Total combinations: {len(MODELS) * len(DATASETS) * len(TASKS)}")
    print("=" * 80)

    print("\nNext steps:")
    print(f"1. Review {args.job_list} to verify job commands")
    print(f"2. Submit jobs: sbatch {args.sbatch_script}")
    print("3. Monitor: squeue -u $USER")
