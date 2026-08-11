#!/usr/bin/env bash
set -euo pipefail

# Conservative recovery when GPU 0 is occupied by an unrelated user. Finish all
# missing checkpoints on owned GPU 1 after the original worker exits, then wait
# for both GPUs before the final complete-grid evaluation.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ORIGINAL_SESSION="${ORIGINAL_SESSION:-archmatrix_full_100}"
STATE_ROOT="${STATE_ROOT:-/dataMeR1/phil/gfm/prodigy-archmatrix/state/icl_arch_matrix}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/icl_arch_matrix_final_recovery}"
mkdir -p "$LOG_ROOT/launch"

while tmux has-session -t "$ORIGINAL_SESSION" 2>/dev/null; do sleep 30; done
while true; do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')"
  printf '%s gpu1=%sMiB\n' "$(date -u +%FT%TZ)" "$used" >> "$LOG_ROOT/launch/gpu1_wait.log"
  (( used <= 2000 )) && break
  sleep 30
done

export PATH="/home/mhchu/miniconda3/bin:$PATH"
STATE_ROOT="$STATE_ROOT" LOG_ROOT="$LOG_ROOT/train" GPUS="1" \
  bash "$SCRIPT_DIR/run_training_tucker.sh"

while true; do
  gpu0="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')"
  gpu1="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')"
  printf '%s gpu0=%sMiB gpu1=%sMiB\n' "$(date -u +%FT%TZ)" "$gpu0" "$gpu1" \
    >> "$LOG_ROOT/launch/eval_gpu_wait.log"
  (( gpu0 <= 2000 && gpu1 <= 2000 )) && break
  sleep 30
done

printf '%s full matrix evaluation started\n' "$(date -u +%FT%TZ)" \
  | tee -a "$LOG_ROOT/launch/matrix_status.log"
STATE_ROOT="$STATE_ROOT" LOG_ROOT="$LOG_ROOT/eval_full" \
  bash "$SCRIPT_DIR/run_evaluation_tucker.sh"
printf '%s full matrix complete\n' "$(date -u +%FT%TZ)" \
  | tee -a "$LOG_ROOT/launch/matrix_status.log"
