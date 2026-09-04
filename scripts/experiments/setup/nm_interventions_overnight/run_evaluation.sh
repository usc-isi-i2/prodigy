#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset CUDA_VISIBLE_DEVICES
python -u scripts/experiments/setup/nm_interventions_overnight/evaluate.py \
  --gpus 1 2 3 --workers-per-gpu 2 "$@"
