#!/usr/bin/env bash
# Frozen two-phase evaluation for the completed final-core training grid.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TRAINING_STATE_ROOT="${TRAINING_STATE_ROOT:-/dataMeR1/phil/gfm/prodigy-final-core/state/final_core}"
EVAL_STATE_ROOT="${EVAL_STATE_ROOT:-${REPO_ROOT}/state/final_core_eval}"
EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${REPO_ROOT}/log/final_core_eval}"
RESULTS_ROOT="${RESULTS_ROOT:-${EVAL_LOG_ROOT}/results}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${EVAL_LOG_ROOT}/summary}"
TRAINING_RUN_STAMP="${TRAINING_RUN_STAMP:-20260807}"
EVALUATION_RUN_STAMP="${EVALUATION_RUN_STAMP:-20260807}"
GPUS_TEXT="${GPUS:-0 1 2 3}"
SEEDS_TEXT="${SEEDS:-0 1 2}"
PHASE="${PHASE:-all}"
DRY_RUN="${DRY_RUN:-0}"
read -r -a GPU_IDS <<< "$GPUS_TEXT"
read -r -a SEED_IDS <<< "$SEEDS_TEXT"

[[ "$PHASE" =~ ^(validation|test|all)$ ]] || { echo "PHASE must be validation, test, or all" >&2; exit 2; }
for gpu in "${GPU_IDS[@]}"; do
  [[ "$gpu" =~ ^[0-3]$ ]] || { echo "refusing non-owned Tucker GPU $gpu" >&2; exit 2; }
done
for seed in "${SEED_IDS[@]}"; do
  [[ "$seed" =~ ^(0|1|2)$ ]] || { echo "seed must be 0, 1, or 2" >&2; exit 2; }
done

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
"$PYTHON" "$SCRIPT_DIR/core_plan.py" > "$PLAN"

jobs=()
while IFS=$'\t' read -r model_id _n_sources sources _aliases; do
  [[ "$model_id" == model_id ]] && continue
  for seed in "${SEED_IDS[@]}"; do jobs+=("${seed}:${model_id}:${sources}"); done
done < "$PLAN"

if [[ "$DRY_RUN" != 1 && ( "$PHASE" == validation || "$PHASE" == all ) ]]; then
  missing=0
  for item in "${jobs[@]}"; do
    IFS=: read -r seed model_id sources <<< "$item"
    for step in 100 300 900 2500; do
      checkpoint="$TRAINING_STATE_ROOT/finalcore_${model_id}_s${seed}_${TRAINING_RUN_STAMP}/checkpoint/state_dict_${step}.ckpt"
      if [[ ! -f "$checkpoint" ]]; then echo "MISSING $checkpoint" >&2; missing=$((missing + 1)); fi
    done
  done
  (( missing == 0 )) || { echo "$missing training checkpoints are missing; evaluation not started" >&2; exit 1; }
fi

run_phase() {
  local phase="$1" worker_count="${#GPU_IDS[@]}" status=0
  local -a pids=()
  worker() {
    local worker_index="$1" gpu="$2" item seed model_id sources index=0
    for item in "${jobs[@]}"; do
      if (( index % worker_count == worker_index )); then
        IFS=: read -r seed model_id sources <<< "$item"
        cmd=("$PYTHON" -u "$SCRIPT_DIR/evaluate_model.py"
             --phase "$phase" --model-id "$model_id" --sources "$sources" --seed "$seed"
             --device 0 --config "$SCRIPT_DIR/training.yaml"
             --training-state-root "$TRAINING_STATE_ROOT"
             --training-run-stamp "$TRAINING_RUN_STAMP"
             --evaluation-state-root "$EVAL_STATE_ROOT"
             --evaluation-log-root "$EVAL_LOG_ROOT/runs"
             --results-root "$RESULTS_ROOT"
             --evaluation-run-stamp "$EVALUATION_RUN_STAMP")
        if [[ "$DRY_RUN" == 1 ]]; then
          printf 'DRY phase=%s gpu=%s' "$phase" "$gpu"; printf ' %q' "${cmd[@]}"; printf '\n'
        else
          echo "[$phase gpu=$gpu] START model=$model_id seed=$seed utc=$(date -u +%FT%TZ)"
          CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$EVAL_LOG_ROOT/queue/${phase}_${model_id}_s${seed}.log" 2>&1
          echo "[$phase gpu=$gpu] DONE model=$model_id seed=$seed utc=$(date -u +%FT%TZ)"
        fi
      fi
      ((index+=1))
    done
  }
  for worker_index in "${!GPU_IDS[@]}"; do
    worker "$worker_index" "${GPU_IDS[$worker_index]}" & pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  return "$status"
}

{
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "training_state_root=$TRAINING_STATE_ROOT"
  echo "training_run_stamp=$TRAINING_RUN_STAMP"
  echo "seeds=$SEEDS_TEXT"
  echo "gpus=$GPUS_TEXT"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$EVAL_LOG_ROOT/provenance.txt"

if [[ "$PHASE" == validation || "$PHASE" == all ]]; then
  run_phase validation
  [[ "$DRY_RUN" == 1 ]] || date -u +%FT%TZ > "$EVAL_LOG_ROOT/validation_complete_utc.txt"
fi
if [[ "$PHASE" == test || "$PHASE" == all ]]; then
  if [[ "$DRY_RUN" != 1 ]]; then
    [[ -f "$EVAL_LOG_ROOT/validation_complete_utc.txt" ]] || {
      echo "test phase is locked until the complete validation phase marker exists" >&2; exit 1;
    }
    missing_selections=0
    for item in "${jobs[@]}"; do
      IFS=: read -r seed model_id sources <<< "$item"
      selection="$RESULTS_ROOT/seed_${seed}/${model_id}/selection.json"
      if [[ ! -f "$selection" ]]; then
        echo "MISSING frozen selection $selection" >&2
        missing_selections=$((missing_selections + 1))
      fi
    done
    (( missing_selections == 0 )) || {
      echo "$missing_selections selections are missing; test remains locked" >&2; exit 1;
    }
  fi
  run_phase test
  if [[ "$DRY_RUN" != 1 ]]; then
    "$PYTHON" "$SCRIPT_DIR/aggregate_results.py" --results-root "$RESULTS_ROOT" --output-root "$SUMMARY_ROOT"
    date -u +%FT%TZ > "$EVAL_LOG_ROOT/test_complete_utc.txt"
  fi
fi
