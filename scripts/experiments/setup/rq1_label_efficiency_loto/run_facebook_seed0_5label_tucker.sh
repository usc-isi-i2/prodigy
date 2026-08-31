#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BASE_STATE="${BASE_STATE:-/dataMeR1/phil/gfm/prodigy-rq1/state/rq1_label_efficiency_loto}"
CHECKPOINT="${CHECKPOINT:-/dataMeR1/phil/gfm/prodigy-nmglobal/state/nm_ladder_global_finalcore/finalcore_ordA_r8_s0_20260814global/state_dict}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_STATE}/facebook_page_category_top30_seed0_5label_v1}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/rq1_label_efficiency_loto/facebook_page_category_top30_seed0_5label_v1}"
CACHE_ROOT="${CACHE_ROOT:-${BASE_STATE}/facebook_subgraph_cache_v1}"

[[ -f "$CHECKPOINT" ]] || { echo "missing all8 checkpoint: $CHECKPOINT" >&2; exit 3; }
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"

for label_seed in 0 1 2 3 4; do
  echo "START facebook model_seed=0 label_seed=$label_seed $(date -Is)"
  SEED=0 LABEL_SEED="$label_seed" GRID_LAYOUT=1 USE_SHARED_CACHE=1 \
    ADAPT_PROTOCOL=revised PROTOCOL_VERSION=facebook-all8-seed0-5label-evalcache-v1 \
    BUDGETS_TEXT="1 10 100 900" \
    TARGETS_TEXT="facebook_page_category_top30" GPUS_TEXT="2 3" SLOTS_PER_GPU=4 \
    PRETRAIN_CHECKPOINT_OVERRIDE="$CHECKPOINT" OUTPUT_ROOT="$OUTPUT_ROOT" \
    LOG_ROOT="$LOG_ROOT" CACHE_ROOT="$CACHE_ROOT" \
    bash "$SCRIPT_DIR/run_adapt_seed_tucker.sh"
  echo "DONE facebook model_seed=0 label_seed=$label_seed $(date -Is)"
done

count="$(find "$OUTPUT_ROOT" -name result.json -type f | wc -l | tr -d ' ')"
[[ "$count" == 40 ]] || { echo "expected 40 Facebook results, found $count" >&2; exit 4; }
echo "FACEBOOK_SEED0_5LABEL_COMPLETE results=$count $(date -Is)"
