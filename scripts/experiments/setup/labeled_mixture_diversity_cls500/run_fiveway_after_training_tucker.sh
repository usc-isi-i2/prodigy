#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_WORKTREE="${TRAIN_WORKTREE:-/dataMeR1/phil/gfm/prodigy-mixconv}"
TRAIN_STATE_ROOT="${TRAIN_STATE_ROOT:-${TRAIN_WORKTREE}/state_labmix500_continuation}"
TRAIN_TMUX="${TRAIN_TMUX:-mixconv}"
EXPECTED_MODELS="${EXPECTED_MODELS:-31}"
BOUNDARY_LOG="${SCRIPT_DIR}/run_logs/fiveway_boundary.log"

mkdir -p "${SCRIPT_DIR}/run_logs"
echo "WAITING_FOR_${EXPECTED_MODELS} $(date)" >> "${BOUNDARY_LOG}"
while [[ "$(find "${TRAIN_STATE_ROOT}" -type f -name state_dict_500.ckpt 2>/dev/null | wc -l)" -lt "${EXPECTED_MODELS}" ]]; do
  sleep 5
done
echo "TRAINING_COMPLETE $(date)" >> "${BOUNDARY_LOG}"

# The original pipeline starts its hard-coded two-way evaluator immediately after
# training. Wait until that boundary, then replace only that tmux session.
old_eval="${TRAIN_WORKTREE}/scripts/experiments/setup/labeled_mixture_diversity_cls500/evaluate.py"
for _attempt in $(seq 1 90); do
  pgrep -f "${old_eval}" >/dev/null && break
  tmux has-session -t "${TRAIN_TMUX}" 2>/dev/null || break
  sleep 1
done

echo "REPLACING_TWO_WAY_EVAL $(date)" >> "${BOUNDARY_LOG}"
tmux kill-session -t "${TRAIN_TMUX}" 2>/dev/null || true
sleep 3
pkill -TERM -f "${old_eval}" 2>/dev/null || true

echo "STARTING_FIVE_WAY $(date)" >> "${BOUNDARY_LOG}"
export PATH="/home/mhchu/miniconda3/bin:${PATH}"
set +e
CONTINUATION_STATE_ROOT="${TRAIN_STATE_ROOT}" \
EVAL_GPUS="${EVAL_GPUS:-0 1 0 1 0}" \
  bash "${SCRIPT_DIR}/run_trajectory_eval_tucker.sh" \
  >> "${SCRIPT_DIR}/run_logs/trajectory_pipeline_5way.log" 2>&1
rc=$?
set -e
echo "FIVE_WAY_FINISHED_RC_${rc} $(date)" >> "${BOUNDARY_LOG}"
exit "${rc}"
