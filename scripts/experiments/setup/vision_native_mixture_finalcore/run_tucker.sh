#!/usr/bin/env bash
# Train the missing seed-0 VISION native feature-similarity mixture rungs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CONFIG="${CONFIG:-${REPO_ROOT}/scripts/experiments/setup/final_core/training.yaml}"
STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state/vision_native_mixture_finalcore}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/vision_native_mixture_finalcore}"
VISION_ROOT="${VISION_ROOT:-/dataMeR1/phil/gfm/upstream/VISION}"
GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
SEED="${SEED:-0}"
for gpu in "$GPU_A" "$GPU_B"; do
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
mkdir -p "$STATE_ROOT" "$LOG_ROOT/train" "$LOG_ROOT/eval" "$LOG_ROOT/results" "$LOG_ROOT/launch"
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

mapfile -t PLAN_ROWS < <(
  "$PYTHON" -m scripts.experiments.setup.vision_native_mixture_finalcore.mixture_plan | tail -n +2
)

{
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "protocol=vision_native_feature_similarity_three_order_mixture"
  echo "orders=A,B,C"
  echo "rungs=1,3,5,7,9"
  echo "unique_models=13"
  echo "new_models=12"
  echo "reused_all9=/dataMeR1/phil/gfm/prodigy-vision-all9/state/vision_all9_finalcore/vision/all9_s0"
  echo "optimizer_updates=2500"
  echo "checkpoint_steps=100,300,900,2500"
  echo "training_seed=$SEED"
  echo "gpus=$GPU_A,$GPU_B"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$LOG_ROOT/launch/provenance.txt"

run_one() {
  local gpu="$1" model_id="$2" sources="$3" checkpoint step result
  [[ "$model_id" != "all9" ]] || return 0
  wait_for_gpu "$gpu"
  checkpoint="$STATE_ROOT/vision/$model_id/checkpoint/state_dict_2500.pt"
  if [[ ! -f "$checkpoint" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m \
      scripts.experiments.setup.vision_all9_finalcore.train_vision_all9 \
      --config "$CONFIG" --upstream-root "$VISION_ROOT" --state-root "$STATE_ROOT" \
      --run-name "$model_id" --model-id "$model_id" --sources "$sources" \
      --steps 2500 --checkpoint-steps 100,300,900,2500 --seed "$SEED" --device 0 \
      > "$LOG_ROOT/train/${model_id}.log" 2>&1
  fi
  [[ -f "$checkpoint" ]] || { echo "missing terminal checkpoint $checkpoint" >&2; return 1; }
  for step in 100 300 900 2500; do
    result="$LOG_ROOT/results/${model_id}_step${step}.jsonl"
    if [[ ! -f "$result" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m \
        scripts.experiments.setup.icl_arch_matrix.evaluate_adapters \
        --architecture vision --upstream-root "$VISION_ROOT" --state-root "$STATE_ROOT" \
        --model-ids "$model_id" --checkpoint-step "$step" --training-seed "$SEED" \
        --eval-episode-seed-offset 0 --include-facebook --results "$result" --device 0 \
        > "$LOG_ROOT/eval/${model_id}_step${step}.log" 2>&1
    fi
    [[ "$(wc -l < "$result")" == 5 ]] || {
      echo "expected five CLS rows in $result" >&2
      return 1
    }
  done
}

worker() {
  local gpu="$1" parity="$2" index=0 row model_id n_sources sources aliases
  for row in "${PLAN_ROWS[@]}"; do
    IFS=$'\t' read -r model_id n_sources sources aliases <<< "$row"
    if [[ "$model_id" == "all9" ]]; then continue; fi
    if (( index % 2 == parity )); then
      run_one "$gpu" "$model_id" "$sources"
    fi
    index=$((index + 1))
  done
}

worker "$GPU_A" 0 & pid_a=$!
worker "$GPU_B" 1 & pid_b=$!
status=0
wait "$pid_a" || status=1
wait "$pid_b" || status=1
[[ "$status" -eq 0 ]] || exit "$status"

expected_files=$((12 * 4))
observed_files="$(find "$LOG_ROOT/results" -maxdepth 1 -type f -name '*.jsonl' | wc -l)"
[[ "$observed_files" -eq "$expected_files" ]] || {
  echo "expected $expected_files new result files, found $observed_files" >&2
  exit 1
}
{
  echo "completed_utc=$(date -u +%FT%TZ)"
  echo "new_models=12"
  echo "new_result_files=$observed_files"
  echo "new_logical_cells=$((observed_files * 5))"
} > "$LOG_ROOT/COMPLETE"
cat "$LOG_ROOT/COMPLETE"
