#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/icl_arch_supervised_target}"
RESULTS_ROOT="${RESULTS_ROOT:-${LOG_ROOT}/results}"
TRAINED_REFERENCE="${TRAINED_REFERENCE:-/dataMeR1/phil/gfm/prodigy-archmatrix-recover1/log/icl_arch_matrix_final_recovery/eval_full/results/summary/classification_long.csv}"
GPU_MLP="${GPU_MLP:-0}"
GPU_GNN="${GPU_GNN:-1}"
DRY_RUN="${DRY_RUN:-0}"

[[ "$GPU_MLP" =~ ^[01]$ ]] || { echo "refusing non-owned Tucker GPU $GPU_MLP" >&2; exit 2; }
[[ "$GPU_GNN" =~ ^[01]$ ]] || { echo "refusing non-owned Tucker GPU $GPU_GNN" >&2; exit 2; }
[[ "$GPU_MLP" != "$GPU_GNN" ]] || { echo "MLP and GNN require distinct owned GPUs" >&2; exit 2; }

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"
mkdir -p "$LOG_ROOT/queue" "$RESULTS_ROOT"
cd "$REPO_ROOT"

mlp_cmd=("$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.evaluate_supervised_target
  --model supervised_mlp --results "$RESULTS_ROOT/supervised_mlp.jsonl" --device 0)
gnn_cmd=("$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.evaluate_supervised_target
  --model supervised_graphsage --results "$RESULTS_ROOT/supervised_graphsage.jsonl" --device 0)
aggregate_cmd=("$PYTHON" -m scripts.experiments.setup.icl_arch_matrix.aggregate_supervised_target
  --mlp "$RESULTS_ROOT/supervised_mlp.jsonl"
  --graphsage "$RESULTS_ROOT/supervised_graphsage.jsonl"
  --trained-reference "$TRAINED_REFERENCE"
  --output-root "$RESULTS_ROOT/summary")

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'DRY CUDA_VISIBLE_DEVICES=%q' "$GPU_MLP"; printf ' %q' "${mlp_cmd[@]}"; printf '\n'
  printf 'DRY CUDA_VISIBLE_DEVICES=%q' "$GPU_GNN"; printf ' %q' "${gnn_cmd[@]}"; printf '\n'
  printf 'DRY'; printf ' %q' "${aggregate_cmd[@]}"; printf '\n'
  exit 0
fi

for gpu in "$GPU_MLP" "$GPU_GNN"; do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
  (( used <= 2000 )) || { echo "GPU $gpu already uses ${used} MiB; refusing to collide" >&2; exit 3; }
done

set +e
CUDA_VISIBLE_DEVICES="$GPU_MLP" "${mlp_cmd[@]}" > "$LOG_ROOT/queue/mlp.log" 2>&1 &
mlp_pid=$!
CUDA_VISIBLE_DEVICES="$GPU_GNN" "${gnn_cmd[@]}" > "$LOG_ROOT/queue/graphsage.log" 2>&1 &
gnn_pid=$!
wait "$mlp_pid"; mlp_status=$?
wait "$gnn_pid"; gnn_status=$?
set -e
(( mlp_status == 0 && gnn_status == 0 )) || {
  echo "supervised worker failure: mlp=$mlp_status graphsage=$gnn_status" >&2
  exit 4
}

"${aggregate_cmd[@]}" > "$LOG_ROOT/queue/aggregate.log" 2>&1
