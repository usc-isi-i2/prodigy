#!/usr/bin/env bash
# Extract frozen representations on owned GPUs 2/3, then run the shared head grid.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/state/adaptation_efficiency}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/adaptation_efficiency}"
TARGETS="${TARGETS:-covid_political,election2020,ukr_rus_suspended,twibot20}"
GPUS_TEXT="${GPUS_TEXT:-2 3}"
read -r -a GPUS <<< "$GPUS_TEXT"
[[ "${#GPUS[@]}" -eq 2 ]] || { echo "exactly two GPUs are required" >&2; exit 2; }
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[23]$ ]] || {
    echo "refusing GPU $gpu: only Tucker GPUs 2 and 3 are authorized" >&2
    exit 2
  }
done

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
PYTHON="${CONDA_PREFIX}/bin/python"
mkdir -p "$OUTPUT_ROOT/caches" "$LOG_ROOT/extract" "$LOG_ROOT/launch"
cd "$REPO_ROOT"

wait_for_gpu() {
  local gpu="$1" used util
  while true; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu")"
    if (( used < 2000 && util < 10 )); then return; fi
    echo "[gpu $gpu] waiting utc=$(date -u +%FT%TZ) used_mib=$used util_pct=$util"
    sleep 60
  done
}

cache_complete() {
  local model="$1" target
  IFS=',' read -r -a target_array <<< "$TARGETS"
  for target in "${target_array[@]}"; do
    [[ -f "$OUTPUT_ROOT/caches/$model/$target.npz" ]] || return 1
  done
}

run_logged() {
  local gpu="$1" name="$2"; shift 2
  if cache_complete "$name"; then
    echo "SKIP complete $name"
    return
  fi
  wait_for_gpu "$gpu"
  echo "START $name gpu=$gpu utc=$(date -u +%FT%TZ)"
  CUDA_VISIBLE_DEVICES="$gpu" "$@" > "$LOG_ROOT/extract/$name.log" 2>&1
  cache_complete "$name" || { echo "incomplete cache for $name" >&2; return 1; }
  echo "DONE $name gpu=$gpu utc=$(date -u +%FT%TZ)"
}

{
  echo "commit=$(git rev-parse HEAD)"
  echo "targets=$TARGETS"
  echo "gpus=${GPUS[*]}"
  echo "label_budgets=0,1,10,100"
  echo "head_updates=0,1,10,100"
  echo "label_seeds=0,1,2"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$LOG_ROOT/launch/provenance.txt"

if ! cache_complete raw_logistic || ! cache_complete raw_mlp; then
  "$PYTHON" -m scripts.experiments.setup.adaptation_efficiency.extract_raw \
    --output-root "$OUTPUT_ROOT/caches" --targets "$TARGETS" \
    > "$LOG_ROOT/extract/raw.log" 2>&1
fi

PRODIGY_ROOT="/dataMeR1/phil/gfm/worktree-runtime-archive-20260812/prodigy-final-core/files/state/final_core"
VISION_STATE="/dataMeR1/phil/gfm/prodigy-vision-all9/state/vision_all9_finalcore/vision"
SAMGPT_ROOT="/dataMeR1/phil/gfm/samgpt-final-core/log/final_core"
SAMGPT_REPO="/dataMeR1/phil/gfm/samgpt-final-core"
SAMGPT_CONFIG="$SAMGPT_REPO/configs/final_core/training.yaml"
GRAPH_SAGE_REPO="/dataMeR1/phil/social-gfm/code"
GRAPH_SAGE_CKPT="/dataMeR1/phil/social-gfm/experiments/pilot-v1/graphsage-all/checkpoint.pt"

worker_prodigy_graphsage() {
  local gpu="$1" seed model checkpoint
  for seed in 0 1 2; do
    model="prodigy_all9_s${seed}"
    checkpoint="$PRODIGY_ROOT/finalcore_all9_s${seed}_20260807/checkpoint/state_dict_2500.ckpt"
    run_logged "$gpu" "$model" "$PYTHON" -m \
      scripts.experiments.setup.adaptation_efficiency.extract_prodigy \
      --checkpoint "$checkpoint" --model-id "$model" --training-seed "$seed" \
      --output-root "$OUTPUT_ROOT/caches" --targets "$TARGETS" --device cuda:0
  done
  run_logged "$gpu" "graphsage_pilot_v1" "$PYTHON" -m \
    scripts.experiments.setup.adaptation_efficiency.extract_graphsage \
    --repository "$GRAPH_SAGE_REPO" --checkpoint "$GRAPH_SAGE_CKPT" \
    --model-id graphsage_pilot_v1 --output-root "$OUTPUT_ROOT/caches" \
    --targets "$TARGETS" --device cuda:0
}

worker_vision_samgpt() {
  local gpu="$1" seed model checkpoint run_dir
  for seed in 0 1 2; do
    model="vision_all9_s${seed}"
    checkpoint="$VISION_STATE/all9_s${seed}/checkpoint/state_dict_2500.pt"
    run_logged "$gpu" "$model" "$PYTHON" -m \
      scripts.experiments.setup.adaptation_efficiency.extract_vision \
      --checkpoint "$checkpoint" --model-id "$model" \
      --output-root "$OUTPUT_ROOT/caches" --targets "$TARGETS" --device cuda:0
  done
  for seed in 39 40 41; do
    model="samgpt_all9_s${seed}"
    run_dir="$SAMGPT_ROOT/seed_${seed}/all9/final_20260807T185324Z"
    checkpoint="$run_dir/checkpoint_update_500.pt"
    run_logged "$gpu" "$model" "$PYTHON" -m \
      scripts.experiments.setup.adaptation_efficiency.extract_samgpt \
      --repository "$SAMGPT_REPO" --training-config "$SAMGPT_CONFIG" \
      --checkpoint "$checkpoint" --resolved-config "$run_dir/resolved_config.json" \
      --model-id "$model" --training-seed "$seed" --output-root "$OUTPUT_ROOT/caches" \
      --targets "$TARGETS" --device cuda:0
  done
}

worker_prodigy_graphsage "${GPUS[0]}" & pid_left=$!
worker_vision_samgpt "${GPUS[1]}" & pid_right=$!
status=0
wait "$pid_left" || status=1
wait "$pid_right" || status=1
[[ "$status" -eq 0 ]] || exit "$status"

RESULTS="$LOG_ROOT/adaptation_cells.csv"
if [[ ! -f "$RESULTS" ]]; then
  cache_args=()
  while IFS= read -r cache; do cache_args+=(--cache "$cache"); done < <(
    find "$OUTPUT_ROOT/caches" -mindepth 2 -maxdepth 2 -type f -name '*.npz' | sort
  )
  "$PYTHON" -m scripts.experiments.setup.adaptation_efficiency.run_head_grid \
    "${cache_args[@]}" --output "$RESULTS" --split-seed 0 --label-seeds 0,1,2
fi

{
  echo "completed_utc=$(date -u +%FT%TZ)"
  echo "results=$RESULTS"
  echo "result_rows=$(($(wc -l < "$RESULTS") - 1))"
} > "$LOG_ROOT/COMPLETE"
cat "$LOG_ROOT/COMPLETE"
