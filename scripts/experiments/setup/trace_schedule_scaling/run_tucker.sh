#!/usr/bin/env bash
# Train TRACE schedule arms. Use PHASE=smoke before PHASE=full.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/training.yaml}"
PHASE="${PHASE:-full}"
GPUS_TEXT="${GPUS:-0 1 2 3}"
DRY_RUN="${DRY_RUN:-0}"

case "$PHASE" in
  smoke)
    TOTAL_STEPS="${TOTAL_STEPS:-20}"
    RUNGS="${RUNGS:-2}"
    SEEDS="${SEEDS:-0}"
    RUN_STAMP="${RUN_STAMP:-20260906smoke}"
    STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state/trace_schedule_scaling_smoke}"
    LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/trace_schedule_scaling_smoke}"
    CHECKPOINT_STEPS="10,20"
    ;;
  full)
    TOTAL_STEPS="${TOTAL_STEPS:-2500}"
    RUNGS="${RUNGS:-2,3,4}"
    SEEDS="${SEEDS:-0,1,2}"
    RUN_STAMP="${RUN_STAMP:-20260906v1}"
    STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state/trace_schedule_scaling}"
    LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/trace_schedule_scaling}"
    CHECKPOINT_STEPS="100,300,625,833,834,900,1250,1666,1667,1875,2500"
    ;;
  *) echo "PHASE must be smoke or full" >&2; exit 2 ;;
esac

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"

read -r -a GPU_IDS <<< "$GPUS_TEXT"
[[ ${#GPU_IDS[@]} -gt 0 ]] || { echo "no GPUs selected" >&2; exit 2; }
for gpu in "${GPU_IDS[@]}"; do
  [[ "$gpu" =~ ^[0-3]$ ]] || {
    echo "GPU $gpu is outside the authorized 0-3 set" >&2
    exit 2
  }
done
[[ "$(git -C "$REPO_ROOT" branch --show-current)" == "codex/trace-schedule-scaling" ]] || {
  echo "wrong branch/worktree: expected codex/trace-schedule-scaling" >&2
  exit 2
}

mkdir -p "$STATE_ROOT" "$LOG_ROOT/train" "$LOG_ROOT/launch"
PLAN="$LOG_ROOT/launch/plan_${RUN_STAMP}.tsv"
cd "$REPO_ROOT"
"$PYTHON" -m scripts.experiments.setup.trace_schedule_scaling.validate_plan \
  --config "$CONFIG" \
  --total-steps "$TOTAL_STEPS" --rungs "$RUNGS" --seeds "$SEEDS" --check-data
"$PYTHON" -m scripts.experiments.setup.trace_schedule_scaling.make_plan \
  --total-steps "$TOTAL_STEPS" \
  --rungs "$RUNGS" --seeds "$SEEDS" > "$PLAN"

jobs=()
while IFS=$'\t' read -r model_id rung seed schedule sources order segment_sources segment_steps source_counts; do
  [[ "$model_id" == model_id ]] && continue
  jobs+=("$model_id|$rung|$seed|$schedule|$sources|$order|$segment_sources|$segment_steps|$source_counts")
done < "$PLAN"
[[ ${#jobs[@]} -gt 0 ]] || { echo "empty plan" >&2; exit 2; }

worker() {
  local worker_index="$1" gpu="$2" index=0 item
  local model_id rung seed schedule sources order segment_sources segment_steps source_counts
  local prefix run_name checkpoint log member_seed source_seed
  for item in "${jobs[@]}"; do
    if (( index % ${#GPU_IDS[@]} == worker_index )); then
      IFS='|' read -r model_id rung seed schedule sources order segment_sources segment_steps source_counts <<< "$item"
      prefix="tracesched_${model_id}"
      run_name="${prefix}_${RUN_STAMP}"
      checkpoint="$STATE_ROOT/$run_name/checkpoint/state_dict_${TOTAL_STEPS}.ckpt"
      log="$LOG_ROOT/train/${run_name}.log"
      member_seed=$((380100 + seed))
      source_seed=$((480100 + seed))
      cmd=("$PYTHON" -u experiments/run_single_experiment.py
        --config "$CONFIG" --device "$gpu" --seed "$seed" --prefix "$prefix"
        --timestamp "$RUN_STAMP" --state_dir "$STATE_ROOT" --log_dir "$LOG_ROOT"
        --dataset_len_cap "$TOTAL_STEPS" --checkpoint_steps "$CHECKPOINT_STEPS"
        --neighbor_sampling_source_subset "$sources"
        --neighbor_sampling_source_schedule "$segment_sources"
        --neighbor_sampling_source_schedule_steps "$segment_steps"
        --neighbor_matching_member_seed "$member_seed"
        --neighbor_sampling_source_schedule_seed "$source_seed")
      if [[ -f "$checkpoint" ]]; then
        echo "[gpu $gpu] SKIP complete $run_name"
      elif [[ -e "$STATE_ROOT/$run_name" ]]; then
        echo "[gpu $gpu] REFUSE incomplete run $STATE_ROOT/$run_name" >&2
        return 1
      elif [[ "$DRY_RUN" == 1 ]]; then
        printf 'DRY train gpu=%s model=%s schedule=%s order=%s counts=%s' \
          "$gpu" "$model_id" "$schedule" "$order" "$source_counts"
        printf ' %q' "${cmd[@]}"
        printf '\n'
      else
        echo "[gpu $gpu] START $model_id order=$order counts=$source_counts utc=$(date -u +%FT%TZ)"
        "${cmd[@]}" > "$log" 2>&1
        [[ -f "$checkpoint" ]] || { echo "missing $checkpoint" >&2; return 1; }
        echo "[gpu $gpu] DONE $model_id utc=$(date -u +%FT%TZ)"
      fi
    fi
    ((index+=1))
  done
}

{
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "phase=$PHASE"
  echo "run_stamp=$RUN_STAMP"
  echo "total_steps=$TOTAL_STEPS"
  echo "rungs=$RUNGS"
  echo "seeds=$SEEDS"
  echo "gpus=$GPUS_TEXT"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$LOG_ROOT/launch/provenance_${RUN_STAMP}.txt"

pids=()
for index in "${!GPU_IDS[@]}"; do
  worker "$index" "${GPU_IDS[$index]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || exit "$status"

MODEL_LIST="$LOG_ROOT/launch/model_list_${RUN_STAMP}.tsv"
printf 'model_id\tcheckpoint\tsources\n' > "$MODEL_LIST"
while IFS=$'\t' read -r model_id _rung _seed _schedule sources _order _segment_sources _segment_steps _source_counts; do
  [[ "$model_id" == model_id ]] && continue
  checkpoint="$STATE_ROOT/tracesched_${model_id}_${RUN_STAMP}/checkpoint/state_dict_${TOTAL_STEPS}.ckpt"
  [[ "$DRY_RUN" == 1 || -f "$checkpoint" ]] || { echo "missing $checkpoint" >&2; exit 1; }
  printf '%s\t%s\t%s\n' "$model_id" "$checkpoint" "$sources" >> "$MODEL_LIST"
done < "$PLAN"
echo "TRACE_SCHEDULE_TRAINING_COMPLETE phase=$PHASE model_list=$MODEL_LIST"
