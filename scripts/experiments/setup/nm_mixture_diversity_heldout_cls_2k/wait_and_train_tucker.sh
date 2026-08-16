#!/usr/bin/env bash
# Wait for an owned Tucker GPU to be stably idle, then launch the full sweep.
# Intended to run inside detached tmux so the caller may disconnect safely.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-0}"
POLL_SECONDS="${POLL_SECONDS:-30}"
FREE_POLLS="${FREE_POLLS:-4}"
MAX_MEMORY_MIB="${MAX_MEMORY_MIB:-1000}"
MAX_UTILIZATION="${MAX_UTILIZATION:-10}"

[[ "${GPU}" =~ ^[01]$ ]] || {
  echo "refusing GPU ${GPU}: this project currently owns only Tucker GPUs 0 and 1" >&2
  exit 2
}
mkdir -p "${SCRIPT_DIR}/run_logs"
exec >>"${SCRIPT_DIR}/run_logs/waiter_gpu${GPU}.log" 2>&1

echo "[$(date --iso-8601=seconds)] waiting for GPU ${GPU}: memory<=${MAX_MEMORY_MIB} MiB and utilization<=${MAX_UTILIZATION}% for ${FREE_POLLS} polls"
streak=0
while (( streak < FREE_POLLS )); do
  IFS=',' read -r memory utilization < <(
    nvidia-smi -i "${GPU}" --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )
  memory="${memory//[[:space:]]/}"
  utilization="${utilization//[[:space:]]/}"
  if (( memory <= MAX_MEMORY_MIB && utilization <= MAX_UTILIZATION )); then
    streak=$((streak + 1))
  else
    streak=0
  fi
  echo "[$(date --iso-8601=seconds)] gpu=${GPU} memory_mib=${memory} util_pct=${utilization} idle_streak=${streak}/${FREE_POLLS}"
  (( streak >= FREE_POLLS )) || sleep "${POLL_SECONDS}"
done

echo "[$(date --iso-8601=seconds)] GPU ${GPU} is stably idle; starting mixture-diversity sweep"
export PATH="/home/mhchu/miniconda3/bin:${PATH}"
GPUS="${GPU}" bash "${SCRIPT_DIR}/run_train_tucker.sh"
