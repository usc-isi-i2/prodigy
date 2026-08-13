#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

export TRAINING_STATE_ROOT="${TRAINING_STATE_ROOT:-${REPO_ROOT}/state/nm_all9_radius_finalcore_10k}"
export EVAL_STATE_ROOT="${EVAL_STATE_ROOT:-${REPO_ROOT}/state/nm_all9_radius_finalcore_10k_eval}"
export EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${REPO_ROOT}/log/nm_all9_radius_finalcore_10k_eval}"
export TRAINING_RUN_STAMP="${TRAINING_RUN_STAMP:-20260812}"
export EVALUATION_RUN_STAMP="${EVALUATION_RUN_STAMP:-20260812}"
export TRAINING_PREFIX="radiusfc10k"
export CHECKPOINT_STEPS="2500 5000 7500 10000"
export SEEDS="0"
export GPUS="${GPUS:-0 1}"
export SLOTS_PER_GPU=1
export PHASE="${PHASE:-all}"

for gpu in $GPUS; do
  [[ "$gpu" =~ ^(0|1)$ ]] || {
    echo "refusing GPU $gpu: current Tucker ownership is GPUs 0 and 1 only" >&2
    exit 2
  }
done

exec bash "$SCRIPT_DIR/run_evaluation_tucker.sh"
