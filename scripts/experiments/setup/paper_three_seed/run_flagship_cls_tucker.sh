#!/usr/bin/env bash
# Target-major fixed 2-way/10-shot classification evaluation for all flagship rungs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_STAMP="${RUN_STAMP:-20260908}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/log/paper_flagship_ladders/${RUN_STAMP}/classification_evaluation}"
SEED0_RUN="${SEED0_RUN:-/dataMeR1/phil/gfm/prodigy-nmi-overnight/log/production/train_20260904_023434}"
REPLICATE_RUN="${REPLICATE_RUN:-${REPO_ROOT}/log/paper_flagship_ladders/${RUN_STAMP}/seeds_1-2}"
REFERENCE_RESULTS="${REFERENCE_RESULTS:-${REPO_ROOT}/scripts/experiments/analysis/transfer/matrices/cross_model/final_core/data/classification_ladder/classification_long.tsv}"
GPUS_TEXT="${GPUS:-0 1 2 3}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
export FINAL_CORE_CPU_THREADS="${FINAL_CORE_CPU_THREADS:-8}"
cd "$REPO_ROOT"

mkdir -p "$OUTPUT_ROOT/results" "$OUTPUT_ROOT/runs" "$OUTPUT_ROOT/eval_state" "$OUTPUT_ROOT/queue"
model_list="$OUTPUT_ROOT/flagship_models.tsv"
"${CONDA_PREFIX}/bin/python" "$SCRIPT_DIR/build_flagship_model_list.py" \
  --run-dirs "$SEED0_RUN" "$REPLICATE_RUN" --output "$model_list"

pids=()
read -r -a gpu_ids <<< "$GPUS_TEXT"
gpu_count="${#gpu_ids[@]}"
(( gpu_count > 0 )) || { echo "no classification GPUs supplied" >&2; exit 2; }
(( WORKERS_PER_GPU > 0 )) || { echo "WORKERS_PER_GPU must be positive" >&2; exit 2; }
worker_count=$((gpu_count * WORKERS_PER_GPU))
for worker in $(seq 0 $((worker_count - 1))); do
  result="$OUTPUT_ROOT/results/worker_${worker}.jsonl"
  queue_log="$OUTPUT_ROOT/queue/worker_${worker}.log"
  [[ ! -e "$result" ]] || { echo "refusing to overwrite $result" >&2; exit 1; }
  physical_gpu="${gpu_ids[$((worker % gpu_count))]}"
  CUDA_VISIBLE_DEVICES="$physical_gpu" "${CONDA_PREFIX}/bin/python" -u \
    -m scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy \
    --config scripts/experiments/setup/final_core/training.yaml \
    --state-root "$OUTPUT_ROOT/unused_managed_state" \
    --eval-state-root "$OUTPUT_ROOT/eval_state/worker_${worker}" \
    --log-root "$OUTPUT_ROOT/runs/worker_${worker}" \
    --results "$result" \
    --run-stamp "$RUN_STAMP" \
    --device 0 \
    --model-list "$model_list" \
    --worker-index "$worker" --worker-count "$worker_count" \
    --include-facebook \
    --reference-results "$REFERENCE_RESULTS" \
    --eval-episode-seed-offset 0 >"$queue_log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if (( status != 0 )); then
  echo "flagship classification evaluation failed" >&2
  exit "$status"
fi

"${CONDA_PREFIX}/bin/python" "$SCRIPT_DIR/aggregate_flagship_cls.py" \
  --input-root "$OUTPUT_ROOT/results" \
  --output "$OUTPUT_ROOT/classification_long.tsv"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUTPUT_ROOT/complete_utc.txt"
