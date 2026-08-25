#!/usr/bin/env bash
# Evaluate the existing VISION native specialists on fixed native pseudo-tasks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
STATE_ROOT="${STATE_ROOT:-/dataMeR1/phil/gfm/prodigy-archnative/state/icl_arch_native_source_900_seed0}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/vision_native_cross_ssl}"
VISION_ROOT="${VISION_ROOT:-/dataMeR1/phil/gfm/upstream/VISION}"
GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
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
mkdir -p "$LOG_ROOT/results" "$LOG_ROOT/logs" "$LOG_ROOT/launch"
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

run_part() {
  local gpu="$1" name="$2" datasets="$3"
  local result="$LOG_ROOT/results/${name}.jsonl"
  [[ -f "$result" ]] && return 0
  wait_for_gpu "$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m \
    scripts.experiments.setup.vision_native_cross_ssl.evaluate \
    --state-root "$STATE_ROOT" --upstream-root "$VISION_ROOT" \
    --datasets "$datasets" --episodes 128 --episode-seed 0 \
    --results "$result" --device 0 > "$LOG_ROOT/logs/${name}.log" 2>&1
}

{
  echo "commit=$(git rev-parse HEAD)"
  echo "protocol=vision_native_feature_similarity_fixed_pseudo_tasks"
  echo "models=5"
  echo "targets=5"
  echo "checkpoints=20,60,100,300,900"
  echo "episodes_per_cell=128"
  echo "training_seed=0"
  echo "episode_seed=0"
  echo "gpus=$GPU_A,$GPU_B"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$LOG_ROOT/launch/provenance.txt"

run_part "$GPU_A" part_a "covid_political,ukr_rus_suspended,facebook_page_reference" & pid_a=$!
run_part "$GPU_B" part_b "election2020,twibot20" & pid_b=$!
status=0
wait "$pid_a" || status=1
wait "$pid_b" || status=1
[[ "$status" -eq 0 ]] || exit "$status"
[[ "$(wc -l < "$LOG_ROOT/results/part_a.jsonl")" -eq 75 ]]
[[ "$(wc -l < "$LOG_ROOT/results/part_b.jsonl")" -eq 50 ]]
{
  echo "completed_utc=$(date -u +%FT%TZ)"
  echo "physical_cells=125"
} > "$LOG_ROOT/COMPLETE"
cat "$LOG_ROOT/COMPLETE"
