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
python -m scripts.experiments.setup.target_performance_mechanisms.run_role_topology \
  --arms scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/data/member_training_verified/arms.json \
  --references scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/data/member_replay_cells.csv \
  --original-roots /dataMeR1/phil/gfm/prodigy-mechanisms/log/target_mechanisms/specialist_cpu_20260906 /dataMeR1/phil/gfm/prodigy-mechanisms/log/target_mechanisms/specialist_cpu_tail_20260906 \
  --fresh-roots /dataMeR1/phil/gfm/prodigy-mechanisms/log/target_mechanisms/fresh_stage_cpu_20260906 \
  "$@"
