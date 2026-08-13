#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/icl_arch_raw_features}"
RESULTS_ROOT="${RESULTS_ROOT:-${LOG_ROOT}/results}"
TRAINED_REFERENCE="${TRAINED_REFERENCE:-/dataMeR1/phil/gfm/prodigy-archmatrix-recover1/log/icl_arch_matrix_final_recovery/eval_full/results/summary/classification_long.csv}"
DRY_RUN="${DRY_RUN:-0}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"
mkdir -p "$LOG_ROOT/queue" "$RESULTS_ROOT"
cd "$REPO_ROOT"

evaluate_cmd=("$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.evaluate_raw_features
  --results "$RESULTS_ROOT/raw_features.jsonl")
aggregate_cmd=("$PYTHON" -m scripts.experiments.setup.icl_arch_matrix.aggregate_raw_features
  --results "$RESULTS_ROOT/raw_features.jsonl"
  --trained-reference "$TRAINED_REFERENCE"
  --output-root "$RESULTS_ROOT/summary")

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'DRY'; printf ' %q' "${evaluate_cmd[@]}"; printf '\n'
  printf 'DRY'; printf ' %q' "${aggregate_cmd[@]}"; printf '\n'
  exit 0
fi

"${evaluate_cmd[@]}" > "$LOG_ROOT/queue/evaluate.log" 2>&1
"${aggregate_cmd[@]}" > "$LOG_ROOT/queue/aggregate.log" 2>&1
