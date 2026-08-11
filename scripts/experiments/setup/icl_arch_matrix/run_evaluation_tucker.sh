#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state/icl_arch_matrix}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/icl_arch_matrix/eval}"
RESULTS_ROOT="${RESULTS_ROOT:-${LOG_ROOT}/results}"
VISION_ROOT="${VISION_ROOT:-/dataMeR1/phil/gfm/upstream/VISION}"
GILT_ROOT="${GILT_ROOT:-/dataMeR1/phil/gfm/upstream/inductnode}"
RUN_STAMP="${RUN_STAMP:-20260810}"
MODEL_IDS="${MODEL_IDS:-}"
DRY_RUN="${DRY_RUN:-0}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"
mkdir -p "$LOG_ROOT/queue" "$RESULTS_ROOT"
cd "$REPO_ROOT"

model_arg=()
[[ -n "$MODEL_IDS" ]] && model_arg=(--model-ids "${MODEL_IDS// /,}")
prodigy_cmd=("$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy
  --state-root "$STATE_ROOT" --log-root "$LOG_ROOT/prodigy_runs"
  --results "$RESULTS_ROOT/prodigy.jsonl" --run-stamp "$RUN_STAMP" --device 0
  "${model_arg[@]}")
vision_cmd=("$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.evaluate_adapters
  --architecture vision --upstream-root "$VISION_ROOT" --state-root "$STATE_ROOT"
  --results "$RESULTS_ROOT/vision.jsonl" --device 0 "${model_arg[@]}")
gilt_cmd=("$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.evaluate_adapters
  --architecture gilt --upstream-root "$GILT_ROOT" --state-root "$STATE_ROOT"
  --results "$RESULTS_ROOT/gilt.jsonl" --device 0 "${model_arg[@]}")

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'DRY CUDA_VISIBLE_DEVICES=0'; printf ' %q' "${prodigy_cmd[@]}"; printf '\n'
  printf 'DRY CUDA_VISIBLE_DEVICES=1'; printf ' %q' "${vision_cmd[@]}"; printf '\n'
  printf 'DRY CUDA_VISIBLE_DEVICES=0'; printf ' %q' "${gilt_cmd[@]}"; printf '\n'
  exit 0
fi

CUDA_VISIBLE_DEVICES=0 "${prodigy_cmd[@]}" > "$LOG_ROOT/queue/prodigy.log" 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES=1 "${vision_cmd[@]}" > "$LOG_ROOT/queue/vision.log" 2>&1 & p1=$!
status=0
wait "$p0" || status=1
wait "$p1" || status=1
(( status == 0 )) || exit "$status"
CUDA_VISIBLE_DEVICES=0 "${gilt_cmd[@]}" > "$LOG_ROOT/queue/gilt.log" 2>&1

if [[ -z "$MODEL_IDS" ]]; then
  "$PYTHON" -m scripts.experiments.setup.icl_arch_matrix.aggregate_results \
    --prodigy "$RESULTS_ROOT/prodigy.jsonl" \
    --vision "$RESULTS_ROOT/vision.jsonl" \
    --gilt "$RESULTS_ROOT/gilt.jsonl" \
    --output-root "$RESULTS_ROOT/summary"
fi
