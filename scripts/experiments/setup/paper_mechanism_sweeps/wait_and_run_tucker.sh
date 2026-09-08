#!/usr/bin/env bash
# Train on GPU 1 as soon as VISION releases it, then serialize the fixed
# evaluation behind the flagship/core audits and use all four owned GPUs.
set -euo pipefail

while tmux has-session -t vision-mixture-seeds 2>/dev/null; do sleep 30; done

stable=0
while (( stable < 4 )); do
  if nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 1 |
      awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1>1000 || $2>10) bad=1} END{exit bad}'; then
    stable=$((stable + 1))
  else
    stable=0
  fi
  sleep 30
done

export PATH="/home/mhchu/miniconda3/bin:$PATH"
PHASE=train GPUS="1" MODELS_PER_GPU=8 WORKER_BUDGET=32 \
  bash "$(dirname "$0")/run_tucker.sh"

EVALUATION_BLOCKERS=(
  paper-optimized-queue paper-onehop-seed2-overlap paper-optimized-relauncher
  paper-flagship-recovery paper-core-fast-eval
)
for session in "${EVALUATION_BLOCKERS[@]}"; do
  while tmux has-session -t "$session" 2>/dev/null; do sleep 30; done
done
stable=0
while (( stable < 4 )); do
  if nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 0,1,2,3 |
      awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1>1000 || $2>10) bad=1} END{exit bad}'; then
    stable=$((stable + 1))
  else
    stable=0
  fi
  sleep 30
done
PHASE=eval GPUS="0 1 2 3" bash "$(dirname "$0")/run_tucker.sh"
