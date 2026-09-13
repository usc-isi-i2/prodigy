#!/usr/bin/env bash
# One-load matched mechanism campaign. PHASE=train can overlap its single-GPU
# training with another campaign; PHASE=eval resumes the exact audited outputs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_STAMP="${RUN_STAMP:-20260908}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/log/paper_mechanism_sweeps/${RUN_STAMP}}"
CONFIG_DIR="${RUN_ROOT}/configs"
TRAIN_DIR="${RUN_ROOT}/training"
NM_DIR="${RUN_ROOT}/nm_evaluation"
CLS_DIR="${RUN_ROOT}/classification_evaluation"
GPUS_TEXT="${GPUS:-0 1 2 3}"
MODELS_PER_GPU="${MODELS_PER_GPU:-5}"
WORKER_BUDGET="${WORKER_BUDGET:-80}"
SEEDS_TEXT="${SEEDS:-0 1 2}"
REFERENCE_RESULTS="${REFERENCE_RESULTS:-${REPO_ROOT}/scripts/experiments/analysis/transfer/matrices/cross_model/final_core/data/classification_ladder/classification_long.tsv}"
PHASE="${PHASE:-all}"
TRAIN_COMPLETE="${RUN_ROOT}/training_complete_utc.txt"
COMPLETE="${RUN_ROOT}/complete_utc.txt"

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
  [[ ! -e "$RUN_ROOT" ]] || { echo "REFUSE existing mechanism root $RUN_ROOT" >&2; exit 1; }
  mkdir -p "$RUN_ROOT"
  "${CONDA_PREFIX}/bin/python" -m scripts.experiments.setup.paper_mechanism_sweeps.plan \
    --output "$CONFIG_DIR"
  mapfile -t configs < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name 'train_*.yaml' | sort)
  [[ ${#configs[@]} -eq 6 ]] || { echo "expected six mechanism configs" >&2; exit 2; }

  "${CONDA_PREFIX}/bin/python" experiments/run_shared_graph.py \
    --configs "${configs[@]}" --seeds $SEEDS_TEXT --gpus $GPUS_TEXT \
    --models-per-gpu "$MODELS_PER_GPU" --worker-budget "$WORKER_BUDGET" \
    --threads-per-model 4 --run-dir "$TRAIN_DIR"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$TRAIN_COMPLETE"
fi

if [[ "$PHASE" == train ]]; then
  exit 0
fi

[[ -f "$TRAIN_COMPLETE" ]] || { echo "missing completed training marker $TRAIN_COMPLETE" >&2; exit 3; }
[[ ! -f "$COMPLETE" ]] || { echo "REFUSE already completed mechanism root $RUN_ROOT" >&2; exit 1; }

"${CONDA_PREFIX}/bin/python" -u -m scripts.experiments.setup.paper_mechanism_sweeps.evaluate \
  --run-dir "$TRAIN_DIR" --output "$NM_DIR" --gpus $GPUS_TEXT \
  --workers-per-gpu 2 --episodes 512

mkdir -p "$CLS_DIR/results" "$CLS_DIR/runs" "$CLS_DIR/eval_state" "$CLS_DIR/queue"
model_list="$CLS_DIR/models.tsv"
"${CONDA_PREFIX}/bin/python" -m scripts.experiments.setup.paper_mechanism_sweeps.models \
  --run-dir "$TRAIN_DIR" --output "$model_list" --exclude-wide
pids=()
read -r -a gpu_ids <<< "$GPUS_TEXT"
worker_count="${#gpu_ids[@]}"
(( worker_count > 0 )) || { echo "no evaluation GPUs supplied" >&2; exit 2; }
for worker in $(seq 0 $((worker_count - 1))); do
  result="$CLS_DIR/results/worker_${worker}.jsonl"
  queue_log="$CLS_DIR/queue/worker_${worker}.log"
  CUDA_VISIBLE_DEVICES="${gpu_ids[$worker]}" "${CONDA_PREFIX}/bin/python" -u \
    -m scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy \
    --config scripts/experiments/setup/final_core/training.yaml \
    --state-root "$CLS_DIR/unused_managed_state" \
    --eval-state-root "$CLS_DIR/eval_state/worker_${worker}" \
    --log-root "$CLS_DIR/runs/worker_${worker}" \
    --results "$result" --run-stamp "$RUN_STAMP" --device 0 \
    --model-list "$model_list" --worker-index "$worker" --worker-count "$worker_count" \
    --include-facebook --reference-results "$REFERENCE_RESULTS" \
    --eval-episode-seed-offset 0 >"$queue_log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || { echo "mechanism classification evaluation failed" >&2; exit "$status"; }
"${CONDA_PREFIX}/bin/python" -m scripts.experiments.setup.paper_mechanism_sweeps.aggregate_cls \
  --input-root "$CLS_DIR/results" --output "$CLS_DIR/classification_long.tsv"
"${CONDA_PREFIX}/bin/python" \
  scripts/experiments/analysis/synthesis/cross_experiment/paper_mechanism_sweeps/analyze.py \
  --nm-root "$NM_DIR" --classification "$CLS_DIR/classification_long.tsv" \
  --output scripts/experiments/analysis/synthesis/cross_experiment/paper_mechanism_sweeps
date -u +%Y-%m-%dT%H:%M:%SZ > "$COMPLETE"
