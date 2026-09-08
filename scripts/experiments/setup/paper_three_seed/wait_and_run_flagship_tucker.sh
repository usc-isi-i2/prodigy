#!/usr/bin/env bash
# Wait for the current owned-GPU jobs, verify a stable idle window, then run flagship ladders.
set -euo pipefail

for session in ${WAIT_SESSIONS:-paper-three-seed-fast gg-ladder-official}; do
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep 30
  done
done

stable=0
while (( stable < 4 )); do
  if nvidia-smi --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits -i 0,1,2,3 |
      awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1>1000 || $2>10) bad=1} END{exit bad}'; then
    stable=$((stable + 1))
  else
    stable=0
  fi
  sleep 30
done

export PATH="/home/mhchu/miniconda3/bin:$PATH"
SEEDS="${SEEDS:-1 2}" GPUS="${GPUS:-0 1 2 3}" \
MODELS_PER_GPU="${MODELS_PER_GPU:-8}" WORKER_BUDGET="${WORKER_BUDGET:-128}" \
RUN_STAMP="${RUN_STAMP:-20260908}" \
bash "$(dirname "$0")/run_flagship_ladders_tucker.sh"
