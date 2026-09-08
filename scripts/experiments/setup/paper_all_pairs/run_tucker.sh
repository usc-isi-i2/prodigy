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
TRAIN_COMPLETE="${RUN_ROOT}/training_complete_utc.txt"
COMPLETE="${RUN_ROOT}/complete_utc.txt"
GPUS_TEXT="${GPUS:-0 1 2 3}"
MODELS_PER_GPU="${MODELS_PER_GPU:-14}"
WORKER_BUDGET="${WORKER_BUDGET:-224}"
SEEDS_TEXT="${SEEDS:-0 1 2}"
PHASE="${PHASE:-all}"
RECOVER_INTERRUPTED="${RECOVER_INTERRUPTED:-0}"

case "$PHASE" in
  all|train|eval) ;;
  *) echo "PHASE must be one of: all, train, eval" >&2; exit 2 ;;
esac

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset CUDA_VISIBLE_DEVICES || true
cd "$REPO_ROOT"

if [[ "$PHASE" != eval ]]; then
  if [[ -e "$RUN_ROOT" && "$RECOVER_INTERRUPTED" != 1 ]]; then
    echo "REFUSE existing all-pairs root $RUN_ROOT" >&2
    exit 1
  fi
  if [[ "$RECOVER_INTERRUPTED" == 1 ]]; then
    [[ -d "$RUN_ROOT" && -f "$TRAIN_DIR/manifest.json" ]] || {
      echo "missing interrupted all-pairs run under $RUN_ROOT" >&2; exit 3;
    }
  else
    mkdir -p "$RUN_ROOT"
    "${CONDA_PREFIX}/bin/python" -m scripts.experiments.setup.paper_all_pairs.plan --output "$CONFIG_DIR"
  fi
  mapfile -t configs < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name 'train_*.yaml' | sort)
  [[ ${#configs[@]} -eq 36 ]] || { echo "expected 36 pair configs" >&2; exit 2; }

  recovery_args=()
  [[ "$RECOVER_INTERRUPTED" != 1 ]] || recovery_args+=(--recover-interrupted)
  "${CONDA_PREFIX}/bin/python" experiments/run_shared_graph.py \
    --configs "${configs[@]}" --seeds $SEEDS_TEXT --gpus $GPUS_TEXT \
    --models-per-gpu "$MODELS_PER_GPU" --worker-budget "$WORKER_BUDGET" \
    --threads-per-model 4 --run-dir "$TRAIN_DIR" "${recovery_args[@]}"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$TRAIN_COMPLETE"
fi

if [[ "$PHASE" == train ]]; then
  exit 0
fi

[[ -f "$TRAIN_COMPLETE" ]] || { echo "missing completed training marker $TRAIN_COMPLETE" >&2; exit 3; }
[[ ! -f "$COMPLETE" ]] || { echo "REFUSE already completed all-pairs root $RUN_ROOT" >&2; exit 1; }

"${CONDA_PREFIX}/bin/python" -u -m scripts.experiments.setup.paper_all_pairs.evaluate \
  --run-dir "$TRAIN_DIR" --output "$EVAL_DIR" \
  --eval-config scripts/experiments/setup/final_core/training.yaml \
  --gpus $GPUS_TEXT --workers-per-gpu 2 --episodes 512
date -u +%Y-%m-%dT%H:%M:%SZ > "$COMPLETE"
