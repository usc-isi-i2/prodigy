#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

while true; do
  mapfile -t used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1)
  if (( ${#used[@]} == 2 && used[0] < 2000 && used[1] < 2000 )); then break; fi
  echo "waiting for GPUs 0,1 utc=$(date -u +%FT%TZ) used_mib=${used[*]:-unknown}"
  sleep 60
done

cd /dataMeR1/phil/gfm/prodigy-archsat900
CONFIG="$PWD/scripts/experiments/setup/icl_arch_matrix/prodigy_training_900.yaml" \
STATE_ROOT="$PWD/state/icl_arch_saturation_900_seed0" \
LOG_ROOT="$PWD/log/icl_arch_saturation_900_seed0" \
RUN_STAMP=20260819 STEPS=900 CHECKPOINTS="20,60,100,300,900" SEEDS_TEXT="0" \
bash scripts/experiments/setup/icl_arch_matrix/run_2000step_three_seed_training_tucker.sh

CONFIG="$PWD/scripts/experiments/setup/icl_arch_matrix/prodigy_training_900.yaml" \
TRAIN_STATE_ROOT="$PWD/state/icl_arch_saturation_900_seed0" \
OUT_ROOT="$PWD/log/icl_arch_saturation_900_seed0_eval" \
RUN_STAMP=20260819 FINAL_STEP=900 CHECKPOINTS_TEXT="20 60 100 300 900" \
CHECKPOINTS_WITH_ZERO="0,20,60,100,300,900" SEEDS_TEXT="0" OFFSETS_TEXT="0" \
bash scripts/experiments/setup/icl_arch_matrix/run_2000step_three_seed_evaluation_tucker.sh

python -u scripts/experiments/analysis/transfer/matrices/cross_architecture/icl_arch_matrix/aggregate_arch_saturation_2000.py \
  --results-root "$PWD/log/icl_arch_saturation_900_seed0_eval/results" \
  --output-root "$PWD/output/icl_arch_saturation_900_seed0" \
  --steps "0,20,60,100,300,900" --training-seeds "0" --eval-offsets "0"
