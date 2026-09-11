#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-node-mlp-ladder}"
export PYTHONPATH=src
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -u -m mixture_scaling.schedule_ladder \
  --state-root "${STATE_ROOT:?set a fresh STATE_ROOT}" \
  --output-root "${RESULTS_ROOT:?set a fresh RESULTS_ROOT}" \
  --log-root "${LOG_ROOT:?set a fresh LOG_ROOT}" \
  --cache-root "${CACHE_ROOT:-/dataMeR1/phil/gfm/mixture-scaling-node-only/state/node_only_transfer/_cache}" \
  --gpus "${GPUS:-0,1,2,3}" "$@"
