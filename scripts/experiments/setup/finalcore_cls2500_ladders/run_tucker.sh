#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
STATE_ROOT="${STATE_ROOT:-/dataMeR1/phil/gfm/prodigy-final-core/state/final_core}"
RUN_STAMP="${RUN_STAMP:-20260807}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/log/finalcore_cls2500_ladders}"
DRY_RUN="${DRY_RUN:-0}"

# The 25 distinct physical checkpoints behind 3 orders x 9 logical rungs:
# three distinct rung-1 specialists, 21 order-specific interior rungs, and all9.
GPU0_MODELS="ss_ukr_rus,ordA_r2,ordA_r3,ordA_r4,ordA_r5,ordA_r6,ordA_r7,ordA_r8,ordB_r2,ordB_r3,ordB_r4,ordB_r5,all9"
GPU1_MODELS="ss_ukr_rus_suspended,ss_twibot20,ordB_r6,ordB_r7,ordB_r8,ordC_r2,ordC_r3,ordC_r4,ordC_r5,ordC_r6,ordC_r7,ordC_r8"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"

mkdir -p "$OUTPUT_ROOT/results" "$OUTPUT_ROOT/runs" "$OUTPUT_ROOT/queue"
cd "$REPO_ROOT"

run_shard() {
  local seed="$1"
  local gpu="$2"
  local shard="$3"
  local models="$4"
  local result="$OUTPUT_ROOT/results/seed${seed}_gpu${gpu}.jsonl"
  local queue_log="$OUTPUT_ROOT/queue/seed${seed}_gpu${gpu}.log"
  local cmd=(
    "$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy
    --config scripts/experiments/setup/final_core/training.yaml
    --state-root "$STATE_ROOT"
    --eval-state-root "$OUTPUT_ROOT/eval_state/seed${seed}_gpu${gpu}"
    --log-root "$OUTPUT_ROOT/runs/seed${seed}_gpu${gpu}"
    --results "$result"
    --run-stamp "$RUN_STAMP"
    --device 0
    --model-ids "$models"
    --include-facebook
    --checkpoint-step 2500
    --checkpoint-layout final-core
    --training-seed "$seed"
    --eval-episode-seed-offset 0
  )
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY seed=%s shard=%s CUDA_VISIBLE_DEVICES=%s' "$seed" "$shard" "$gpu"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  if [[ -e "$result" ]]; then
    echo "refusing to overwrite $result" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$queue_log" 2>&1
}

for seed in 0 1 2; do
  run_shard "$seed" 0 left "$GPU0_MODELS" & p0=$!
  run_shard "$seed" 1 right "$GPU1_MODELS" & p1=$!
  status=0
  wait "$p0" || status=1
  wait "$p1" || status=1
  if (( status != 0 )); then
    echo "classification evaluation failed for training seed $seed" >&2
    exit "$status"
  fi
done

if [[ "$DRY_RUN" == 1 ]]; then
  exit 0
fi

"$PYTHON" "$SCRIPT_DIR/aggregate.py" \
  --input-root "$OUTPUT_ROOT/results" \
  --output "$OUTPUT_ROOT/classification_long.tsv"
