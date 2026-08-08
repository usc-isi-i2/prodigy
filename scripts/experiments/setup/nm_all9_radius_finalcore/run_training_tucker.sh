#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state/nm_all9_radius_finalcore}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/nm_all9_radius_finalcore}"
FEASIBILITY_REPORT="${FEASIBILITY_REPORT:-${LOG_ROOT}/preflight/feasibility.json}"
GPUS_TEXT="${GPUS:-0 1 2 3}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-1}"
SEEDS_TEXT="${SEEDS:-0 1 2}"
RUN_STAMP="${RUN_STAMP:-20260807}"
MODE="${MODE:-train}"
DRY_RUN="${DRY_RUN:-0}"
read -r -a GPU_IDS <<< "$GPUS_TEXT"
read -r -a SEED_IDS <<< "$SEEDS_TEXT"

[[ "$MODE" =~ ^(smoke|train)$ ]] || { echo "MODE must be smoke or train" >&2; exit 2; }
[[ "$SLOTS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || { echo "SLOTS_PER_GPU must be positive" >&2; exit 2; }
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
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"

mkdir -p "$STATE_ROOT" "$LOG_ROOT/train" "$LOG_ROOT/launch"
PLAN="$LOG_ROOT/launch/plan.tsv"
cd "$REPO_ROOT"
"$PYTHON" "$SCRIPT_DIR/radius_plan.py" > "$PLAN"
[[ "$(($(wc -l < "$PLAN") - 1))" == 3 ]] || { echo "plan must contain three arms" >&2; exit 2; }

if [[ "$DRY_RUN" != 1 ]]; then
  [[ -f "$FEASIBILITY_REPORT" ]] || {
    echo "missing feasibility gate $FEASIBILITY_REPORT; run run_preflight_tucker.sh first" >&2
    exit 1
  }
  "$PYTHON" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); assert p.get("ready") is True, "feasibility report is not ready"' \
    "$FEASIBILITY_REPORT"
fi

jobs=()
while IFS=$'\t' read -r arm_id config _radii; do
  [[ "$arm_id" == arm_id ]] && continue
  if [[ "$MODE" == smoke ]]; then
    jobs+=("0:${arm_id}:${config}")
  else
    for seed in "${SEED_IDS[@]}"; do jobs+=("${seed}:${arm_id}:${config}"); done
  fi
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

worker() {
  local worker_index="$1" gpu="$2" item seed arm_id config index=0
  for item in "${jobs[@]}"; do
    if (( index % worker_count == worker_index )); then
      IFS=: read -r seed arm_id config <<< "$item"
      if [[ "$MODE" == smoke ]]; then
        prefix="radiusfc_smoke_${arm_id}_s${seed}"
        expected_step=5
        extra=(--dataset_len_cap 5 --epochs 1 --checkpoint_steps 5 --workers 0)
      else
        prefix="radiusfc_${arm_id}_s${seed}"
        expected_step=2500
        extra=()
      fi
      run_name="${prefix}_${RUN_STAMP}"
      checkpoint="$STATE_ROOT/$run_name/checkpoint/state_dict_${expected_step}.ckpt"
      if [[ -f "$checkpoint" ]]; then
        echo "[gpu $gpu] SKIP complete $run_name"
        ((index+=1)); continue
      fi
      if [[ -e "$STATE_ROOT/$run_name" ]]; then
        echo "[gpu $gpu] REFUSE incomplete existing run $STATE_ROOT/$run_name" >&2
        return 1
      fi
      cmd=("$PYTHON" -u experiments/run_single_experiment.py
           --config "$config" --device "$gpu" --seed "$seed"
           --prefix "$prefix" --timestamp "$RUN_STAMP"
           --state_dir "$STATE_ROOT" --log_dir "$LOG_ROOT" "${extra[@]}")
      if [[ "$DRY_RUN" == 1 ]]; then
        printf 'DRY mode=%s gpu=%s' "$MODE" "$gpu"; printf ' %q' "${cmd[@]}"; printf '\n'
      else
        echo "[gpu $gpu] START arm=$arm_id seed=$seed mode=$MODE utc=$(date -u +%FT%TZ)"
        "${cmd[@]}" > "$LOG_ROOT/train/${run_name}.log" 2>&1
        [[ -f "$checkpoint" ]] || { echo "missing terminal checkpoint $checkpoint" >&2; return 1; }
        echo "[gpu $gpu] DONE $run_name utc=$(date -u +%FT%TZ)"
      fi
    fi
    ((index+=1))
  done
}

{
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "mode=$MODE"
  echo "seeds=$SEEDS_TEXT"
  echo "gpus=$GPUS_TEXT"
  echo "slots_per_gpu=$SLOTS_PER_GPU"
  echo "feasibility_report=$FEASIBILITY_REPORT"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$LOG_ROOT/launch/provenance_${MODE}.txt"

pids=()
for gpu_index in "${!GPU_IDS[@]}"; do
  for ((slot=0; slot<SLOTS_PER_GPU; slot++)); do
    worker_index=$((gpu_index * SLOTS_PER_GPU + slot))
    worker "$worker_index" "${GPU_IDS[$gpu_index]}" & pids+=("$!")
  done
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
