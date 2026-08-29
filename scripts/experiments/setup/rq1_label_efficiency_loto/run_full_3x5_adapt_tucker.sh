#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BASE_STATE="${BASE_STATE:-/dataMeR1/phil/gfm/prodigy-rq1/state/rq1_label_efficiency_loto}"
PRETRAIN_STATE_ROOT="${PRETRAIN_STATE_ROOT:-${BASE_STATE}/pretrain}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_STATE}/adapt_full_3x5_evalcache_v1}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/rq1_label_efficiency_loto/adapt_full_3x5_evalcache_v1}"
CACHE_ROOT="${CACHE_ROOT:-${BASE_STATE}/subgraph_cache_v2}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$CACHE_ROOT"

for model_seed in 0 1 2; do
  for label_seed in 0 1 2 3 4; do
    echo "START model_seed=$model_seed label_seed=$label_seed $(date -Is)"
    SEED="$model_seed" LABEL_SEED="$label_seed" GRID_LAYOUT=1 \
      USE_SHARED_CACHE=1 ADAPT_PROTOCOL=revised \
      PROTOCOL_VERSION=full-3x5-evalcache-patience3-v1 \
      GPUS_TEXT="2 3" SLOTS_PER_GPU=8 \
      PRETRAIN_STATE_ROOT="$PRETRAIN_STATE_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" \
      LOG_ROOT="$LOG_ROOT" CACHE_ROOT="$CACHE_ROOT" \
      bash "$SCRIPT_DIR/run_adapt_seed_tucker.sh"
    echo "DONE model_seed=$model_seed label_seed=$label_seed $(date -Is)"
  done
done

count="$(find "$OUTPUT_ROOT" -name result.json -type f | wc -l | tr -d " ")"
[[ "$count" == 480 ]] || { echo "expected 480 results, found $count" >&2; exit 4; }
echo "FULL_3X5_COMPLETE results=$count $(date -Is)"
