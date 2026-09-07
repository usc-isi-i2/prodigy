#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BASE_STATE="${BASE_STATE:-/dataMeR1/phil/gfm/prodigy-rq1/state/rq1_label_efficiency_loto}"
BASE_LOG="${BASE_LOG:-${REPO_ROOT}/log/rq1_label_efficiency_loto}"
CACHE_ROOT="${CACHE_ROOT:-${BASE_STATE}/subgraph_cache_v2}"

SEED=2 CONFIG="${SCRIPT_DIR}/pretrain_revised_seed2.yaml" \
  STATE_ROOT="${BASE_STATE}/pretrain_revised_seed2" \
  LOG_ROOT="${BASE_LOG}/pretrain_revised_seed2" \
  bash "${SCRIPT_DIR}/run_pretrain_seed_tucker.sh"

SEED=2 ADAPT_PROTOCOL=revised \
  PRETRAIN_STATE_ROOT="${BASE_STATE}/pretrain_revised_seed2" \
  OUTPUT_ROOT="${BASE_STATE}/adapt_revised_seed2" \
  LOG_ROOT="${BASE_LOG}/adapt_revised_seed2" \
  CACHE_ROOT="$CACHE_ROOT" \
  bash "${SCRIPT_DIR}/run_adapt_seed_tucker.sh"
