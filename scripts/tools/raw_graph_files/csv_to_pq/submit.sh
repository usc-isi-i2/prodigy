#!/bin/bash
#SBATCH --job-name=covid_parquet
#SBATCH --partition=main
#SBATCH --array=0-99
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/covid_parquet_%A_%a.out
#SBATCH --error=logs/covid_parquet_%A_%a.err

set -euo pipefail
mkdir -p logs

module purge
# >>> activate your python env here <
# e.g.: source ~/envs/parquet/bin/activate
# or:   module load python/3.11 && source ~/venv/bin/activate

python convert.py \
    --task-id ${SLURM_ARRAY_TASK_ID} \
    --num-tasks 100