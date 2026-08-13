#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TRAINING_STATE_ROOT="${TRAINING_STATE_ROOT:-${REPO_ROOT}/state/nm_all9_radius_finalcore}"
EVAL_STATE_ROOT="${EVAL_STATE_ROOT:-${REPO_ROOT}/state/nm_all9_radius_finalcore_eval}"
EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${REPO_ROOT}/log/nm_all9_radius_finalcore_eval}"
RESULTS_ROOT="${RESULTS_ROOT:-${EVAL_LOG_ROOT}/results}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${EVAL_LOG_ROOT}/summary}"
TRAINING_RUN_STAMP="${TRAINING_RUN_STAMP:-20260807}"
TRAINING_PREFIX="${TRAINING_PREFIX:-radiusfc}"
CHECKPOINT_STEPS_TEXT="${CHECKPOINT_STEPS:-100 300 900 2500}"
EVALUATION_RUN_STAMP="${EVALUATION_RUN_STAMP:-20260807}"
GPUS_TEXT="${GPUS:-0 1 2 3}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-1}"
SEEDS_TEXT="${SEEDS:-0 1 2}"
PHASE="${PHASE:-all}"
DRY_RUN="${DRY_RUN:-0}"
VALIDATION_MODE="${VALIDATION_MODE:-shared}"
EVAL_BATCH_COUNT="${EVAL_BATCH_COUNT:-}"
EVAL_WORKERS="${EVAL_WORKERS:-}"
read -r -a GPU_IDS <<< "$GPUS_TEXT"
read -r -a SEED_IDS <<< "$SEEDS_TEXT"
read -r -a CHECKPOINT_STEP_IDS <<< "$CHECKPOINT_STEPS_TEXT"

[[ "$PHASE" =~ ^(validation|test|all)$ ]] || { echo "PHASE must be validation, test, or all" >&2; exit 2; }
[[ "$VALIDATION_MODE" =~ ^(shared|legacy)$ ]] || { echo "VALIDATION_MODE must be shared or legacy" >&2; exit 2; }
[[ "$SLOTS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || { echo "SLOTS_PER_GPU must be positive" >&2; exit 2; }
[[ -z "$EVAL_BATCH_COUNT" || "$EVAL_BATCH_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "EVAL_BATCH_COUNT must be positive" >&2; exit 2; }
[[ -z "$EVAL_WORKERS" || "$EVAL_WORKERS" =~ ^[0-9]+$ ]] || { echo "EVAL_WORKERS must be non-negative" >&2; exit 2; }
for gpu in "${GPU_IDS[@]}"; do
  [[ "$gpu" =~ ^[0-3]$ ]] || { echo "refusing non-owned Tucker GPU $gpu" >&2; exit 2; }
done
for seed in "${SEED_IDS[@]}"; do
  [[ "$seed" =~ ^(0|1|2)$ ]] || { echo "seed must be 0, 1, or 2" >&2; exit 2; }
done
for step in "${CHECKPOINT_STEP_IDS[@]}"; do
  [[ "$step" =~ ^[1-9][0-9]*$ ]] || { echo "checkpoint steps must be positive integers" >&2; exit 2; }
done
CHECKPOINT_STEPS_CSV="$(IFS=,; echo "${CHECKPOINT_STEP_IDS[*]}")"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"

mkdir -p "$EVAL_STATE_ROOT" "$EVAL_LOG_ROOT/queue" "$RESULTS_ROOT" "$SUMMARY_ROOT"
PLAN="$EVAL_LOG_ROOT/plan.tsv"
cd "$REPO_ROOT"
"$PYTHON" "$SCRIPT_DIR/radius_plan.py" > "$PLAN"

jobs=()
while IFS=$'\t' read -r arm_id _config _radii; do
  [[ "$arm_id" == arm_id ]] && continue
  for seed in "${SEED_IDS[@]}"; do jobs+=("${seed}:${arm_id}"); done
done < "$PLAN"
worker_count=$(( ${#GPU_IDS[@]} * SLOTS_PER_GPU ))

if [[ "$DRY_RUN" != 1 ]]; then
  for gpu in "${GPU_IDS[@]}"; do
    used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    (( used <= 1000 )) || { echo "GPU $gpu is busy (${used} MiB); refusing launch" >&2; exit 1; }
  done
  available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  required_gib=$((worker_count * 125 + 100))
  required_kib=$((required_gib * 1024 * 1024))
  (( available_kib >= required_kib )) || {
    echo "insufficient host RAM for $worker_count graph loads: need ${required_gib} GiB available" >&2
    exit 1
  }
fi

if [[ "$DRY_RUN" != 1 && ( "$PHASE" == validation || "$PHASE" == all ) ]]; then
  missing=0
  for item in "${jobs[@]}"; do
    IFS=: read -r seed arm_id <<< "$item"
    for step in "${CHECKPOINT_STEP_IDS[@]}"; do
      checkpoint="$TRAINING_STATE_ROOT/${TRAINING_PREFIX}_${arm_id}_s${seed}_${TRAINING_RUN_STAMP}/checkpoint/state_dict_${step}.ckpt"
      if [[ ! -f "$checkpoint" ]]; then echo "MISSING $checkpoint" >&2; missing=$((missing + 1)); fi
    done
  done
  (( missing == 0 )) || { echo "$missing checkpoints are missing; evaluation not started" >&2; exit 1; }
fi

run_phase() {
  local phase="$1" status=0
  local -a pids=()
  worker() {
    local worker_index="$1" gpu="$2" item seed arm_id index=0
    for item in "${jobs[@]}"; do
      if (( index % worker_count == worker_index )); then
        IFS=: read -r seed arm_id <<< "$item"
        cmd=("$PYTHON" -u "$SCRIPT_DIR/evaluate_radius.py"
             --phase "$phase" --arm "$arm_id" --seed "$seed" --device 0
             --training-state-root "$TRAINING_STATE_ROOT"
             --training-run-stamp "$TRAINING_RUN_STAMP"
             --training-prefix "$TRAINING_PREFIX"
             --checkpoint-steps "$CHECKPOINT_STEPS_CSV"
             --evaluation-state-root "$EVAL_STATE_ROOT"
             --evaluation-log-root "$EVAL_LOG_ROOT/runs"
             --results-root "$RESULTS_ROOT"
             --evaluation-run-stamp "$EVALUATION_RUN_STAMP"
             --validation-mode "$VALIDATION_MODE")
        [[ -z "$EVAL_BATCH_COUNT" ]] || cmd+=(--eval-batch-count "$EVAL_BATCH_COUNT")
        [[ -z "$EVAL_WORKERS" ]] || cmd+=(--workers "$EVAL_WORKERS")
        if [[ "$DRY_RUN" == 1 ]]; then
          printf 'DRY phase=%s gpu=%s' "$phase" "$gpu"; printf ' %q' "${cmd[@]}"; printf '\n'
        else
          echo "[$phase gpu=$gpu] START arm=$arm_id seed=$seed utc=$(date -u +%FT%TZ)"
          CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$EVAL_LOG_ROOT/queue/${phase}_${arm_id}_s${seed}.log" 2>&1
          echo "[$phase gpu=$gpu] DONE arm=$arm_id seed=$seed utc=$(date -u +%FT%TZ)"
        fi
      fi
      ((index+=1))
    done
  }
  for gpu_index in "${!GPU_IDS[@]}"; do
    for ((slot=0; slot<SLOTS_PER_GPU; slot++)); do
      worker_index=$((gpu_index * SLOTS_PER_GPU + slot))
      worker "$worker_index" "${GPU_IDS[$gpu_index]}" & pids+=("$!")
    done
  done
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  return "$status"
}

if [[ "$PHASE" == validation || "$PHASE" == all ]]; then
  run_phase validation
  [[ "$DRY_RUN" == 1 ]] || date -u +%FT%TZ > "$EVAL_LOG_ROOT/validation_complete_utc.txt"
fi
if [[ "$PHASE" == test || "$PHASE" == all ]]; then
  if [[ "$DRY_RUN" != 1 ]]; then
    [[ -f "$EVAL_LOG_ROOT/validation_complete_utc.txt" ]] || {
      echo "test locked until complete validation marker exists" >&2; exit 1;
    }
    for item in "${jobs[@]}"; do
      IFS=: read -r seed arm_id <<< "$item"
      [[ -f "$RESULTS_ROOT/seed_${seed}/${arm_id}/selection.json" ]] || {
        echo "missing frozen selection for $arm_id seed $seed" >&2; exit 1;
      }
    done
  fi
  run_phase test
  if [[ "$DRY_RUN" != 1 ]]; then
    "$PYTHON" "$SCRIPT_DIR/aggregate_radius_results.py" \
      --results-root "$RESULTS_ROOT" --output-root "$SUMMARY_ROOT" \
      --seeds "${SEEDS_TEXT// /,}"
    date -u +%FT%TZ > "$EVAL_LOG_ROOT/test_complete_utc.txt"
  fi
fi
