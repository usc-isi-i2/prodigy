#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
STATE_ROOT="${STATE_ROOT:-/dataMeR1/phil/gfm/prodigy-radiusfc/state/nm_all9_radius_finalcore}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/log/nm_all9_radius_finalcore_cls}"
RUN_STAMP="${RUN_STAMP:-20260807}"
CHECKPOINT_STEPS_TEXT="${CHECKPOINT_STEPS:-100 300 900 2500}"
SEEDS_TEXT="${SEEDS:-0 1 2}"
GPUS_TEXT="${GPUS:-0 1}"
DRY_RUN="${DRY_RUN:-0}"
read -r -a GPU_IDS <<< "$GPUS_TEXT"

(( ${#GPU_IDS[@]} == 1 || ${#GPU_IDS[@]} == 2 )) || {
  echo "GPUS must contain one or two owned GPU ids" >&2
  exit 2
}
for gpu in "${GPU_IDS[@]}"; do
  [[ "$gpu" =~ ^(0|1)$ ]] || { echo "refusing non-owned GPU $gpu" >&2; exit 2; }
done

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"

mkdir -p "$OUTPUT_ROOT/results" "$OUTPUT_ROOT/runs" "$OUTPUT_ROOT/eval_state" "$OUTPUT_ROOT/queue"
cd "$REPO_ROOT"

run_shard() {
  local seed="$1" step="$2" gpu="$3" models="$4"
  local shard="seed${seed}_step${step}_gpu${gpu}"
  local result="$OUTPUT_ROOT/results/${shard}.jsonl"
  local -a cmd=(
    "$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy
    --config scripts/experiments/setup/nm_all9_radius_finalcore/global.yaml
    --state-root "$STATE_ROOT"
    --eval-state-root "$OUTPUT_ROOT/eval_state/$shard"
    --log-root "$OUTPUT_ROOT/runs/$shard"
    --results "$result"
    --run-stamp "$RUN_STAMP"
    --device 0
    --model-ids "$models"
    --include-facebook
    --checkpoint-step "$step"
    --checkpoint-layout radius-finalcore
    --training-seed "$seed"
    --eval-episode-seed-offset 0
  )
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY CUDA_VISIBLE_DEVICES=%s' "$gpu"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return
  fi
  [[ ! -e "$result" ]] || { echo "refusing to overwrite $result" >&2; return 1; }
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$OUTPUT_ROOT/queue/${shard}.log" 2>&1
}

for seed in $SEEDS_TEXT; do
  for step in $CHECKPOINT_STEPS_TEXT; do
    if (( ${#GPU_IDS[@]} == 1 )); then
      run_shard "$seed" "$step" "${GPU_IDS[0]}" "global,radius_mix,close_only"
    else
      run_shard "$seed" "$step" "${GPU_IDS[0]}" "global,radius_mix" & p0=$!
      run_shard "$seed" "$step" "${GPU_IDS[1]}" "close_only" & p1=$!
      status=0
      wait "$p0" || status=1
      wait "$p1" || status=1
      (( status == 0 )) || exit "$status"
    fi
  done
done

echo "CLASSIFICATION_TRAJECTORIES_COMPLETE results=$OUTPUT_ROOT/results"
