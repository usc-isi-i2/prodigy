#!/usr/bin/env bash
# Refill the six-per-GPU slots released by the one-hop overlap while the old
# remainder is live. Fall back to GPU 1 if that overlap window has closed.
# Fixed evaluation remains serialized behind every primary audit.
set -euo pipefail

export PATH="/home/mhchu/miniconda3/bin:$PATH"
while tmux has-session -t paper-onehop-seed2-overlap 2>/dev/null; do sleep 30; done
if tmux has-session -t paper-optimized-queue 2>/dev/null; then
  # This restores the already measured two-load envelope of 14 trainers per
  # device: eight remainder trainers plus six mechanism trainers.
  PHASE=train GPUS="0 2 3" MODELS_PER_GPU=6 WORKER_BUDGET=72 \
    bash "$(dirname "$0")/run_tucker.sh"
else
  # The primary handoff may already own 0/2/3. Wait for VISION and use GPU 1.
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
  PHASE=train GPUS="1" MODELS_PER_GPU=8 WORKER_BUDGET=32 \
    bash "$(dirname "$0")/run_tucker.sh"
fi

EVALUATION_BLOCKERS=(
  paper-optimized-queue paper-onehop-seed2-overlap paper-optimized-relauncher
  paper-flagship-recovery paper-core-fast-eval vision-mixture-seeds
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
