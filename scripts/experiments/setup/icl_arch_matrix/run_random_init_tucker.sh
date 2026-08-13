#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
DEVICE="${DEVICE:-1}"
[[ "$DEVICE" =~ ^[01]$ ]] || { echo "refusing non-owned Tucker GPU $DEVICE" >&2; exit 2; }

STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state/icl_arch_random_init}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/icl_arch_random_init}"
RESULTS_ROOT="${RESULTS_ROOT:-${LOG_ROOT}/results}"
VISION_ROOT="${VISION_ROOT:-/dataMeR1/phil/gfm/upstream/VISION}"
GILT_ROOT="${GILT_ROOT:-/dataMeR1/phil/gfm/upstream/inductnode}"
TRAINED_REFERENCE="${TRAINED_REFERENCE:-/dataMeR1/phil/gfm/prodigy-archmatrix-recover1/log/icl_arch_matrix_final_recovery/eval_full/results/summary/classification_long.csv}"
RUN_STAMP="${RUN_STAMP:-20260811_random_init}"
DRY_RUN="${DRY_RUN:-0}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"
mkdir -p "$STATE_ROOT" "$LOG_ROOT/queue" "$RESULTS_ROOT"
cd "$REPO_ROOT"

prodigy_cmd=("$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy
  --random-init --state-root "$STATE_ROOT" --log-root "$LOG_ROOT/prodigy_runs"
  --eval-state-root "$STATE_ROOT/prodigy_eval" --results "$RESULTS_ROOT/prodigy.jsonl"
  --run-stamp "$RUN_STAMP" --device 0)
vision_cmd=("$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.evaluate_adapters
  --random-init --architecture vision --upstream-root "$VISION_ROOT"
  --results "$RESULTS_ROOT/vision.jsonl" --device 0)
gilt_cmd=("$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.evaluate_adapters
  --random-init --architecture gilt --upstream-root "$GILT_ROOT"
  --results "$RESULTS_ROOT/gilt.jsonl" --device 0)

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'DRY CUDA_VISIBLE_DEVICES=%q' "$DEVICE"; printf ' %q' "${prodigy_cmd[@]}"; printf '\n'
  printf 'DRY CUDA_VISIBLE_DEVICES=%q' "$DEVICE"; printf ' %q' "${vision_cmd[@]}"; printf '\n'
  printf 'DRY CUDA_VISIBLE_DEVICES=%q' "$DEVICE"; printf ' %q' "${gilt_cmd[@]}"; printf '\n'
  exit 0
fi

used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$DEVICE" | tr -d ' ')"
(( used <= 2000 )) || { echo "GPU $DEVICE already uses ${used} MiB; refusing to collide" >&2; exit 3; }

CUDA_VISIBLE_DEVICES="$DEVICE" "${prodigy_cmd[@]}" > "$LOG_ROOT/queue/prodigy.log" 2>&1
CUDA_VISIBLE_DEVICES="$DEVICE" "${vision_cmd[@]}" > "$LOG_ROOT/queue/vision.log" 2>&1
CUDA_VISIBLE_DEVICES="$DEVICE" "${gilt_cmd[@]}" > "$LOG_ROOT/queue/gilt.log" 2>&1

"$PYTHON" -m scripts.experiments.setup.icl_arch_matrix.aggregate_random_init \
  --prodigy "$RESULTS_ROOT/prodigy.jsonl" \
  --vision "$RESULTS_ROOT/vision.jsonl" \
  --gilt "$RESULTS_ROOT/gilt.jsonl" \
  --trained-reference "$TRAINED_REFERENCE" \
  --output-root "$RESULTS_ROOT/summary"
