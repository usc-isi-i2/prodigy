#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/training.yaml}"
STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state/final_core}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/final_core}"
GPUS_TEXT="${GPUS:-0 1 2 3}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"
SEEDS_TEXT="${SEEDS:-0 1 2}"
RUN_STAMP="${RUN_STAMP:-20260807}"
DRY_RUN="${DRY_RUN:-0}"
read -r -a GPU_IDS <<< "$GPUS_TEXT"
read -r -a SEED_IDS <<< "$SEEDS_TEXT"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"
mkdir -p "$STATE_ROOT" "$LOG_ROOT/train" "$LOG_ROOT/launch"
PLAN="$LOG_ROOT/launch/plan.tsv"
cd "$REPO_ROOT"
"$PYTHON" "$SCRIPT_DIR/core_plan.py" > "$PLAN"
[[ "$(($(wc -l < "$PLAN") - 1))" == 31 ]] || { echo "plan must contain 31 models" >&2; exit 2; }

jobs=()
while IFS=$'\t' read -r model_id _n_sources sources _aliases; do
  [[ "$model_id" == model_id ]] && continue
  for seed in "${SEED_IDS[@]}"; do jobs+=("${seed}:${model_id}:${sources}"); done
done < "$PLAN"
worker_count=$(( ${#GPU_IDS[@]} * SLOTS_PER_GPU ))

worker() {
  local worker_index="$1" gpu="$2" item seed model_id sources index=0
  for item in "${jobs[@]}"; do
    if (( index % worker_count == worker_index )); then
      IFS=: read -r seed model_id sources <<< "$item"
      prefix="finalcore_${model_id}_s${seed}"
      run_name="${prefix}_${RUN_STAMP}"
      checkpoint="$STATE_ROOT/$run_name/checkpoint/state_dict_2500.ckpt"
      if [[ -f "$checkpoint" ]]; then
        echo "[gpu $gpu] SKIP complete $run_name"
        ((index+=1)); continue
      fi
      if [[ -e "$STATE_ROOT/$run_name" ]]; then
        echo "[gpu $gpu] REFUSE incomplete existing run $STATE_ROOT/$run_name" >&2
        return 1
      fi
      cmd=("$PYTHON" -u experiments/run_single_experiment.py --config "$CONFIG"
           --device "$gpu" --seed "$seed" --prefix "$prefix" --timestamp "$RUN_STAMP"
           --state_dir "$STATE_ROOT" --log_dir "$LOG_ROOT"
           --neighbor_sampling_source_subset "$sources")
      if [[ "$DRY_RUN" == 1 ]]; then
        printf 'DRY gpu=%s' "$gpu"; printf ' %q' "${cmd[@]}"; printf '\n'
      else
        echo "[gpu $gpu] START model=$model_id seed=$seed utc=$(date -u +%FT%TZ)"
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
  echo "seeds=$SEEDS_TEXT"
  echo "gpus=$GPUS_TEXT"
  echo "slots_per_gpu=$SLOTS_PER_GPU"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$LOG_ROOT/launch/provenance.txt"

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
