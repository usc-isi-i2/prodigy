#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_STAMP="${RUN_STAMP:-20260813}"
GPUS="${GPUS:-0}"
DRY_RUN="${DRY_RUN:-0}"

export ARM_IDS="distance_stratified"
export RUN_STAMP
export GPUS
export DRY_RUN
export STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state/nm_all9_distance_stratified_10k}"
export LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/nm_all9_distance_stratified_10k}"
export FEASIBILITY_REPORT="${FEASIBILITY_REPORT:-${REPO_ROOT}/log/nm_all9_distance_stratified/preflight/feasibility.json}"

echo "[pipeline] distance-stratified 10k training starts utc=$(date -u +%FT%TZ)"
bash "$SCRIPT_DIR/run_convergence_10k_tucker.sh"

echo "[pipeline] training complete; four-panel evaluation starts utc=$(date -u +%FT%TZ)"
export TRAINING_STATE_ROOT="$STATE_ROOT"
export EVAL_STATE_ROOT="${EVAL_STATE_ROOT:-${REPO_ROOT}/state/nm_all9_distance_stratified_10k_eval}"
export EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${REPO_ROOT}/log/nm_all9_distance_stratified_10k_eval}"
export TRAINING_RUN_STAMP="$RUN_STAMP"
export EVALUATION_RUN_STAMP="$RUN_STAMP"
export TRAINING_PREFIX="radiusfc10k"
export CHECKPOINT_STEPS="2500 5000 7500 10000"
export SEEDS="0"
export SLOTS_PER_GPU=1
export PHASE=all
bash "$SCRIPT_DIR/run_evaluation_tucker.sh"
echo "[pipeline] complete utc=$(date -u +%FT%TZ)"
