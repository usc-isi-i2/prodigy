#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/icl_arch_matrix}"
PILOT_MODEL="${PILOT_MODEL:-ss_ukr_rus}"
GPU_FREE_MIB="${GPU_FREE_MIB:-2000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "$LOG_ROOT/launch"
cd "$REPO_ROOT"

while true; do
  gpu0_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')"
  gpu1_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')"
  printf '%s gpu0=%sMiB gpu1=%sMiB\n' \
    "$(date -u +%FT%TZ)" "$gpu0_mib" "$gpu1_mib" >> "$LOG_ROOT/launch/gpu_wait.log"
  if (( gpu0_mib <= GPU_FREE_MIB && gpu1_mib <= GPU_FREE_MIB )); then
    break
  fi
  sleep "$POLL_SECONDS"
done

printf '%s corrected pilot training started\n' "$(date -u +%FT%TZ)" | tee -a "$LOG_ROOT/launch/matrix_status.log"
MODEL_IDS="$PILOT_MODEL" bash "$SCRIPT_DIR/run_training_tucker.sh"

pilot_eval_root="$LOG_ROOT/eval_pilot"
if [[ ! -f "$pilot_eval_root/results/summary/summary.json" ]]; then
  printf '%s corrected pilot evaluation started\n' "$(date -u +%FT%TZ)" | tee -a "$LOG_ROOT/launch/matrix_status.log"
  MODEL_IDS="$PILOT_MODEL" LOG_ROOT="$pilot_eval_root" \
    bash "$SCRIPT_DIR/run_evaluation_tucker.sh"
fi
printf '%s corrected pilot hash gate passed\n' "$(date -u +%FT%TZ)" | tee -a "$LOG_ROOT/launch/matrix_status.log"

printf '%s full matrix training started\n' "$(date -u +%FT%TZ)" | tee -a "$LOG_ROOT/launch/matrix_status.log"
bash "$SCRIPT_DIR/run_training_tucker.sh"

full_eval_root="$LOG_ROOT/eval_full"
if [[ ! -f "$full_eval_root/results/summary/summary.json" ]]; then
  printf '%s full matrix evaluation started\n' "$(date -u +%FT%TZ)" | tee -a "$LOG_ROOT/launch/matrix_status.log"
  LOG_ROOT="$full_eval_root" bash "$SCRIPT_DIR/run_evaluation_tucker.sh"
fi
printf '%s full matrix complete\n' "$(date -u +%FT%TZ)" | tee -a "$LOG_ROOT/launch/matrix_status.log"
