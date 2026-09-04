#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset CUDA_VISIBLE_DEVICES
python -m unittest discover -s scripts/experiments/setup/nm_interventions_overnight/tests -v
python -u experiments/run_shared_graph.py \
 --configs scripts/experiments/setup/nm_interventions_overnight/configs/*_r8_s0.yaml \
 --gpus 0 1 2 3 --models-per-gpu 4 --worker-budget 64 --threads-per-model 2 \
 --run-dir /dataMeR1/phil/gfm/prodigy-nmi-overnight/log/smoke_v1 \
 --smoke-steps 20 -- --campaign_eval_interval 20 --campaign_val_per_source 1
