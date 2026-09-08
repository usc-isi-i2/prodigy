#!/usr/bin/env bash
# Train and evaluate every two-source subset after the primary paper queues finish.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_STAMP="${RUN_STAMP:-20260908}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/log/paper_all_pairs/${RUN_STAMP}}"
TRAIN_DIR="${RUN_ROOT}/training"
CONFIG_DIR="${RUN_ROOT}/configs"
EVAL_DIR="${RUN_ROOT}/nm_evaluation"
GPUS_TEXT="${GPUS:-0 1 2 3}"
MODELS_PER_GPU="${MODELS_PER_GPU:-14}"
WORKER_BUDGET="${WORKER_BUDGET:-224}"
SEEDS_TEXT="${SEEDS:-0 1 2}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset CUDA_VISIBLE_DEVICES || true
cd "$REPO_ROOT"

if [[ -e "$RUN_ROOT" ]]; then
  echo "REFUSE existing all-pairs root $RUN_ROOT" >&2
  exit 1
fi
mkdir -p "$RUN_ROOT"
"${CONDA_PREFIX}/bin/python" -m scripts.experiments.setup.paper_all_pairs.plan --output "$CONFIG_DIR"
mapfile -t configs < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name 'train_*.yaml' | sort)
[[ ${#configs[@]} -eq 36 ]] || { echo "expected 36 pair configs" >&2; exit 2; }

"${CONDA_PREFIX}/bin/python" experiments/run_shared_graph.py \
  --configs "${configs[@]}" --seeds $SEEDS_TEXT --gpus $GPUS_TEXT \
  --models-per-gpu "$MODELS_PER_GPU" --worker-budget "$WORKER_BUDGET" \
  --threads-per-model 4 --run-dir "$TRAIN_DIR"

"${CONDA_PREFIX}/bin/python" -u -m scripts.experiments.setup.paper_all_pairs.evaluate \
  --run-dir "$TRAIN_DIR" --output "$EVAL_DIR" \
  --eval-config scripts/experiments/setup/final_core/training.yaml \
  --gpus $GPUS_TEXT --workers-per-gpu 2 --episodes 512
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/complete_utc.txt"
