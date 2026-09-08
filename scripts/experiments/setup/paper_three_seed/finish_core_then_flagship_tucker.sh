#!/usr/bin/env bash
# Preserve the live seed-1 batch, stop the reload-heavy legacy queue, then run optimized batches.
set -euo pipefail

LIVE_STATUS="${LIVE_STATUS:-/dataMeR1/phil/gfm/prodigy-paper-fast/log/paper_three_seed_fast/20260908/seed1_ladder_1hop/status.json}"
LIVE_SESSION="${LIVE_SESSION:-paper-three-seed-fast}"

while true; do
  if [[ -f "$LIVE_STATUS" ]]; then
    if grep -q '"status": "complete"' "$LIVE_STATUS"; then
      break
    fi
    if grep -q '"status": "failed_or_interrupted"' "$LIVE_STATUS"; then
      echo "live seed-1 batch failed; refusing dependent launch" >&2
      exit 1
    fi
  fi
  sleep 30
done

# The old parent would reload the 111 GB graph for every remaining family and seed.
# Interrupt it only after the completed seed-1 batch has its terminal status file.
if tmux has-session -t "$LIVE_SESSION" 2>/dev/null; then
  tmux send-keys -t "$LIVE_SESSION" C-c
fi
while tmux has-session -t "$LIVE_SESSION" 2>/dev/null; do sleep 5; done
while tmux has-session -t gg-ladder-official 2>/dev/null; do sleep 30; done

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
GPUS="0 1 2 3" MODELS_PER_GPU=8 WORKER_BUDGET=128 RUN_STAMP=20260908 \
  bash "$(dirname "$0")/run_fast_remainder_tucker.sh"

SEEDS="1 2" GPUS="0 1 2 3" MODELS_PER_GPU=8 WORKER_BUDGET=128 RUN_STAMP=20260908 \
  bash "$(dirname "$0")/run_flagship_ladders_tucker.sh"
