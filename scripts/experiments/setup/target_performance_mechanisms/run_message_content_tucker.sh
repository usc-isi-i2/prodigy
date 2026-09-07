#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=""
export WANDB_MODE=offline
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python -m scripts.experiments.setup.target_performance_mechanisms.run_message_content \
  --reference-replay /dataMeR1/phil/gfm/prodigy-role-topology/log/role_topology_full_20260907 \
  "$@"
