#!/usr/bin/env bash
# Refill slots released by the one-hop overlap while remaining on GPUs 2--3.
# Fixed evaluation remains serialized behind every primary audit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_STAMP="${RUN_STAMP:-20260908}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/log/paper_mechanism_sweeps/${RUN_STAMP}}"
TRAIN_COMPLETE="${RUN_ROOT}/training_complete_utc.txt"
GPUS_TEXT="${GPUS:-}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
if [[ ! -f "$TRAIN_COMPLETE" ]]; then
  while tmux has-session -t paper-onehop-seed2-overlap 2>/dev/null; do sleep 30; done
  if tmux has-session -t paper-optimized-queue 2>/dev/null; then
    # This restores the already measured two-load envelope of 14 trainers per
    # device: eight remainder trainers plus six mechanism trainers.
    PHASE=train GPUS="2 3" MODELS_PER_GPU=6 WORKER_BUDGET=48 \
      bash "$SCRIPT_DIR/run_tucker.sh"
  else
    # Wait until both campaign devices are stable rather than spilling to 0/1.
    stable=0
    while (( stable < 4 )); do
      if nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 2,3 |
          awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1>1000 || $2>10) bad=1} END{exit bad}'; then
        stable=$((stable + 1))
      else
        stable=0
      fi
      sleep 30
    done
    PHASE=train GPUS="2 3" MODELS_PER_GPU=8 WORKER_BUDGET=64 \
      bash "$SCRIPT_DIR/run_tucker.sh"
  fi
fi

EVALUATION_BLOCKERS=(
  paper-optimized-queue paper-onehop-seed2-overlap paper-optimized-relauncher
  paper-flagship-recovery paper-core-fast-eval
)
for session in "${EVALUATION_BLOCKERS[@]}"; do
  while tmux has-session -t "$session" 2>/dev/null; do sleep 30; done
done

if [[ -z "$GPUS_TEXT" ]]; then
  while true; do
    available=()
    for gpu in 2 3; do
      values="$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits | tr -d ' ')"
      IFS=, read -r used util <<< "$values"
      if (( used < 1000 && util < 10 )); then available+=("$gpu"); fi
    done
    if (( ${#available[@]} >= 2 )); then
      GPUS_TEXT="${available[*]}"
      break
    fi
    sleep 30
  done
fi
read -r -a gpu_ids <<< "$GPUS_TEXT"
(( ${#gpu_ids[@]} >= 2 )) || { echo "mechanism evaluation requires at least two GPUs" >&2; exit 2; }
for gpu in "${gpu_ids[@]}"; do
  [[ "$gpu" =~ ^[23]$ ]] || { echo "refusing GPU $gpu: this campaign is restricted to 2-3" >&2; exit 2; }
done
[[ "${gpu_ids[*]}" == "2 3" || "${gpu_ids[*]}" == "3 2" ]] \
  || { echo "mechanism evaluation requires exactly GPUs 2 and 3" >&2; exit 2; }
stable=0
while (( stable < 4 )); do
  gpu_csv="$(tr ' ' ',' <<< "$GPUS_TEXT")"
  if nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu_csv" |
      awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1>1000 || $2>10) bad=1} END{exit bad}'; then
    stable=$((stable + 1))
  else
    stable=0
  fi
  sleep 30
done
PHASE=eval GPUS="$GPUS_TEXT" CLS_WORKERS_PER_GPU=2 bash "$SCRIPT_DIR/run_tucker.sh"
