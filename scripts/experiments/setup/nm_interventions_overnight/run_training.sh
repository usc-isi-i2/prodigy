#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset CUDA_VISIBLE_DEVICES
python -u scripts/experiments/setup/nm_interventions_overnight/queue.py \
 --root /dataMeR1/phil/gfm/prodigy-nmi-overnight/log/production \
 --gpus ${GPUS:-1 2 3} --models-per-gpu ${MODELS_PER_GPU:-4} "$@"
