#!/usr/bin/env bash
# Train six schedule arms with one dynamic queue per GPU, then evaluate CLS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/training.yaml}"
PLAN="${PLAN:-${SCRIPT_DIR}/plan.tsv}"
STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state/nm_two_source_schedule_pilot}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/nm_two_source_schedule_pilot}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
GPUS_TEXT="${GPUS:-0 1 2 3}"
DRY_RUN="${DRY_RUN:-0}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONDONTWRITEBYTECODE=1
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"

read -r -a GPU_IDS <<< "$GPUS_TEXT"
[[ ${#GPU_IDS[@]} -gt 0 ]] || { echo "no GPUs selected" >&2; exit 2; }
for gpu in "${GPU_IDS[@]}"; do
  [[ "$gpu" =~ ^[0-3]$ ]] || { echo "GPU $gpu is outside the authorized 0-3 set" >&2; exit 2; }
done

mkdir -p "$STATE_ROOT" "$LOG_ROOT/train" "$LOG_ROOT/eval" "$LOG_ROOT/launch"
cd "$REPO_ROOT"
"$PYTHON" "$SCRIPT_DIR/validate_plan.py" --check-data

jobs=()
while IFS=$'\t' read -r model_id pair schedule sources sequence sequence_steps; do
  [[ "$model_id" == model_id ]] && continue
  jobs+=("$model_id|$pair|$schedule|$sources|$sequence|$sequence_steps")
done < "$PLAN"
[[ ${#jobs[@]} == 6 ]] || { echo "expected six jobs, found ${#jobs[@]}" >&2; exit 2; }

worker() {
  local worker_index="$1" gpu="$2" index=0 item
  local model_id pair schedule sources sequence sequence_steps prefix run_name checkpoint log
  for item in "${jobs[@]}"; do
    if (( index % ${#GPU_IDS[@]} == worker_index )); then
      IFS='|' read -r model_id pair schedule sources sequence sequence_steps <<< "$item"
      prefix="nm2sched_${model_id}_s0"
      run_name="${prefix}_${RUN_STAMP}"
      checkpoint="$STATE_ROOT/$run_name/checkpoint/state_dict_2500.ckpt"
      log="$LOG_ROOT/train/${run_name}.log"
      cmd=("$PYTHON" -u experiments/run_single_experiment.py
        --config "$CONFIG" --device "$gpu" --seed 0 --prefix "$prefix"
        --timestamp "$RUN_STAMP" --state_dir "$STATE_ROOT" --log_dir "$LOG_ROOT"
        --neighbor_sampling_source_subset "$sources")
      if [[ "$schedule" == sequential ]]; then
        cmd+=(--neighbor_sampling_source_sequence "$sequence"
              --neighbor_sampling_source_sequence_steps "$sequence_steps")
      fi
      if [[ -f "$checkpoint" ]]; then
        echo "[gpu $gpu] SKIP complete $run_name"
      elif [[ -e "$STATE_ROOT/$run_name" ]]; then
        echo "[gpu $gpu] REFUSE incomplete run $STATE_ROOT/$run_name" >&2
        return 1
      elif [[ "$DRY_RUN" == 1 ]]; then
        printf 'DRY train gpu=%s' "$gpu"; printf ' %q' "${cmd[@]}"; printf '\n'
      else
        echo "[gpu $gpu] START $model_id utc=$(date -u +%FT%TZ)"
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
  echo "run_stamp=$RUN_STAMP"
  echo "gpus=$GPUS_TEXT"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$LOG_ROOT/launch/provenance_${RUN_STAMP}.txt"

pids=()
for index in "${!GPU_IDS[@]}"; do
  worker "$index" "${GPU_IDS[$index]}" & pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || exit "$status"

MODEL_LIST="$LOG_ROOT/launch/model_list_${RUN_STAMP}.txt"
: > "$MODEL_LIST"
while IFS=$'\t' read -r model_id _pair _schedule _sources _sequence _sequence_steps; do
  [[ "$model_id" == model_id ]] && continue
  checkpoint="$STATE_ROOT/nm2sched_${model_id}_s0_${RUN_STAMP}/checkpoint/state_dict_2500.ckpt"
  if [[ "$DRY_RUN" != 1 && ! -f "$checkpoint" ]]; then
    echo "missing checkpoint $checkpoint" >&2
    exit 1
  fi
  printf '%s %s\n' "$model_id" "$checkpoint" >> "$MODEL_LIST"
done < "$PLAN"

eval_cmd=("$PYTHON" -u scripts/eval/eval_ckpts_all_graph_tasks_tucker.py
  --model-list "$MODEL_LIST" --data-root /dataMeR1/phil/data
  --datasets covid_political,election2020,ukr_rus_suspended,twibot20
  --tasks classification --shots 10 --workers 2 --continue-on-error
  --gpus "$(IFS=,; echo "${GPU_IDS[*]}")" --seed 0
  -- --n_hop 2
  --neighbor_sampling_hop_sizes 9,9 --neighbor_sampling_node_limit 101
  --neighbor_matching_walk_hops 1 --log_dir "$LOG_ROOT/eval" --state_dir "$STATE_ROOT/eval")

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'DRY eval'; printf ' %q' "${eval_cmd[@]}"; printf '\n'
  exit 0
fi
"${eval_cmd[@]}" | tee "$LOG_ROOT/launch/eval_${RUN_STAMP}.log"
echo "NM_TWO_SOURCE_SCHEDULE_PILOT_COMPLETE run_stamp=$RUN_STAMP log_root=$LOG_ROOT"
