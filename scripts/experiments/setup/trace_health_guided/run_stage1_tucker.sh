#!/usr/bin/env bash
# Train matched-budget source specialists and the naive merged baseline.
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
    SOURCES="${SOURCES:-ukr_rus,covid}"
    SEEDS="${SEEDS:-0}"
    RUN_STAMP="${RUN_STAMP:-20260907smoke}"
    STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state/trace_health_guided_smoke}"
    LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/trace_health_guided_smoke}"
    ;;
  full)
    TOTAL_STEPS="${TOTAL_STEPS:-2500}"
    SOURCES="${SOURCES:-ukr_rus,covid,midterm,covid_political}"
    SEEDS="${SEEDS:-0,1,2}"
    RUN_STAMP="${RUN_STAMP:-20260907v1}"
    STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state/trace_health_guided}"
    LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/trace_health_guided}"
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
  [[ "$gpu" =~ ^[0-3]$ ]] || { echo "unauthorized GPU $gpu" >&2; exit 2; }
done
[[ "$(git -C "$REPO_ROOT" branch --show-current)" == "codex/trace-health-guided" ]] || {
  echo "wrong branch/worktree: expected codex/trace-health-guided" >&2
  exit 2
}

mkdir -p "$STATE_ROOT" "$LOG_ROOT/train" "$LOG_ROOT/launch"
PLAN="$LOG_ROOT/launch/stage1_plan_${RUN_STAMP}.tsv"
cd "$REPO_ROOT"
"$PYTHON" -m scripts.experiments.setup.trace_health_guided.make_stage1_plan \
  --sources "$SOURCES" --seeds "$SEEDS" > "$PLAN"

jobs=()
while IFS=$'\t' read -r model_id condition seed sources; do
  [[ "$model_id" == model_id ]] && continue
  jobs+=("$model_id|$condition|$seed|$sources")
done < "$PLAN"
[[ ${#jobs[@]} -gt 0 ]] || { echo "empty plan" >&2; exit 2; }

worker() {
  local worker_index="$1" gpu="$2" index=0 item
  local model_id condition seed sources prefix run_name checkpoint log member_seed source_seed
  local -a cmd
  for item in "${jobs[@]}"; do
    if (( index % ${#GPU_IDS[@]} == worker_index )); then
      IFS='|' read -r model_id condition seed sources <<< "$item"
      prefix="tracehg_${model_id}"
      run_name="${prefix}_${RUN_STAMP}"
      checkpoint="$STATE_ROOT/$run_name/checkpoint/state_dict_${TOTAL_STEPS}.ckpt"
      log="$LOG_ROOT/train/${run_name}.log"
      member_seed=$((380100 + seed))
      cmd=("$PYTHON" -u experiments/run_single_experiment.py
        --config "$CONFIG" --device "$gpu" --seed "$seed" --prefix "$prefix"
        --timestamp "$RUN_STAMP" --state_dir "$STATE_ROOT" --log_dir "$LOG_ROOT"
        --dataset_len_cap "$TOTAL_STEPS" --checkpoint_steps "$TOTAL_STEPS"
        --neighbor_sampling_source_subset "$sources"
        --neighbor_matching_member_seed "$member_seed")
      if [[ "$condition" == single ]]; then
        source_seed=$((480100 + seed))
        cmd+=(--neighbor_sampling_episode_source graph_id
          --neighbor_sampling_source_schedule "$sources"
          --neighbor_sampling_source_schedule_steps "$TOTAL_STEPS"
          --neighbor_sampling_source_schedule_seed "$source_seed")
      fi
      if [[ -f "$checkpoint" ]]; then
        echo "[gpu $gpu] SKIP complete $run_name"
      elif [[ -e "$STATE_ROOT/$run_name" ]]; then
        echo "[gpu $gpu] REFUSE incomplete run $STATE_ROOT/$run_name" >&2
        return 1
      elif [[ "$DRY_RUN" == 1 ]]; then
        printf 'DRY train gpu=%s model=%s condition=%s' "$gpu" "$model_id" "$condition"
        printf ' %q' "${cmd[@]}"
        printf '\n'
      else
        echo "[gpu $gpu] START $model_id condition=$condition utc=$(date -u +%FT%TZ)"
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
  echo "sources=$SOURCES"
  echo "seeds=$SEEDS"
  echo "gpus=$GPUS_TEXT"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$LOG_ROOT/launch/stage1_provenance_${RUN_STAMP}.txt"

pids=()
for index in "${!GPU_IDS[@]}"; do
  worker "$index" "${GPU_IDS[$index]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || exit "$status"

MODEL_LIST="$LOG_ROOT/launch/stage1_model_list_${RUN_STAMP}.tsv"
printf 'model_id\tcheckpoint\tsources\n' > "$MODEL_LIST"
while IFS=$'\t' read -r model_id _condition _seed sources; do
  [[ "$model_id" == model_id ]] && continue
  checkpoint="$STATE_ROOT/tracehg_${model_id}_${RUN_STAMP}/checkpoint/state_dict_${TOTAL_STEPS}.ckpt"
  [[ "$DRY_RUN" == 1 || -f "$checkpoint" ]] || { echo "missing $checkpoint" >&2; exit 1; }
  printf '%s\t%s\t%s\n' "$model_id" "$checkpoint" "$sources" >> "$MODEL_LIST"
done < "$PLAN"
echo "TRACE_HEALTH_STAGE1_COMPLETE model_list=$MODEL_LIST"
