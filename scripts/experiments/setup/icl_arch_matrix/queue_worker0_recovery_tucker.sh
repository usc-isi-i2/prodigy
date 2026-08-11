#!/usr/bin/env bash
set -euo pipefail

# Recover exactly the jobs assigned to worker 0 after its ordA_r8 VISION OOM.
# The complementary worker-1 jobs may continue concurrently on GPU 1.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
STATE_ROOT="${STATE_ROOT:-/dataMeR1/phil/gfm/prodigy-archmatrix/state/icl_arch_matrix}"
BASE_LOG_ROOT="${BASE_LOG_ROOT:-${REPO_ROOT}/log/icl_arch_matrix_worker0_recovery}"
GPU_FREE_MIB="${GPU_FREE_MIB:-2000}"
mkdir -p "$(dirname "$BASE_LOG_ROOT")"

while true; do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')"
  printf '%s gpu0=%sMiB\n' "$(date -u +%FT%TZ)" "$used" >> "${BASE_LOG_ROOT}_gpu_wait.log"
  (( used <= GPU_FREE_MIB )) && break
  sleep 30
done

jobs=(
  vision:ordA_r8
  prodigy:all9 gilt:all9
  vision:ordB_r2
  prodigy:ordB_r3 gilt:ordB_r3
  vision:ordB_r4
  prodigy:ordB_r5 gilt:ordB_r5
  vision:ordB_r6
  prodigy:ordB_r7 gilt:ordB_r7
  vision:ordB_r8
  prodigy:ordC_r2 gilt:ordC_r2
  vision:ordC_r3
  prodigy:ordC_r4 gilt:ordC_r4
  vision:ordC_r5
  prodigy:ordC_r6 gilt:ordC_r6
  vision:ordC_r7
  prodigy:ordC_r8 gilt:ordC_r8
)

export PATH="/home/mhchu/miniconda3/bin:$PATH"
for item in "${jobs[@]}"; do
  architecture="${item%%:*}"
  model_id="${item#*:}"
  LOG_ROOT="$BASE_LOG_ROOT/${architecture}_${model_id}" \
  STATE_ROOT="$STATE_ROOT" GPUS="0" ARCHITECTURES="$architecture" MODEL_IDS="$model_id" \
    bash "$SCRIPT_DIR/run_training_tucker.sh"
done
