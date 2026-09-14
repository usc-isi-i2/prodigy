#!/usr/bin/env bash
# Fill the four source-only rows in the pilot-v1 fixed-compute CLS matrix.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${ROOT}/configs/social_sources.yaml}"
STATE_ROOT="${STATE_ROOT:-${ROOT}/state/social_source_matrix_s0}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/social_source_matrix_s0/raw}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/log/social_source_matrix_s0}"
GPUS_TEXT="${GPUS_TEXT:-2 3}"
read -r -a GPUS <<< "$GPUS_TEXT"
SOURCES=(covid cp_hk midterm ukr_rus)
TARGETS=(covid_political election2020 facebook_page_reference twibot20 ukr_rus_suspended)

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
PYTHON="${CONDA_PREFIX}/bin/python"
mkdir -p "$STATE_ROOT" "$RESULT_ROOT" "$LOG_ROOT/train" "$LOG_ROOT/eval" "$LOG_ROOT/launch"
cd "$ROOT"

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

run_source() {
  local gpu="$1" source="$2" run_id checkpoint target output
  run_id="matrix_source_${source}_w256_existing_s0"
  checkpoint="$STATE_ROOT/$run_id/checkpoints/step_2500.pt"
  wait_for_gpu "$gpu"
  if [[ ! -f "$checkpoint" ]]; then
    if [[ -e "$STATE_ROOT/$run_id" ]]; then
      echo "refusing ambiguous partial run: $STATE_ROOT/$run_id" >&2
      return 1
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m mixture_scaling.train \
      --config "$CONFIG" --sources "$source" --run-id "$run_id" \
      --device 0 --seed 0 --max-steps 2500 --output-root "$STATE_ROOT" \
      > "$LOG_ROOT/train/${run_id}.log" 2>&1
  fi
  [[ -f "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; return 1; }

  for target in "${TARGETS[@]}"; do
    output="$RESULT_ROOT/${source}_to_${target}_step2500.json"
    if [[ ! -f "$output" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m mixture_scaling.evaluate \
        --config "$CONFIG" --checkpoint "$checkpoint" --target "$target" \
        --device 0 --output "$output" \
        > "$LOG_ROOT/eval/${source}_to_${target}_step2500.log" 2>&1
    fi
  done
}

worker() {
  local worker_index="$1" gpu="$2" index=0 source
  for source in "${SOURCES[@]}"; do
    if (( index % ${#GPUS[@]} == worker_index )); then
      run_source "$gpu" "$source"
    fi
    index=$((index + 1))
  done
}

{
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "protocol=pilot-v1-link-prediction-fixed-compute"
  echo "checkpoint_step=2500"
  echo "training_seed=0"
  echo "labels_per_class=10"
  echo "sources=${SOURCES[*]}"
  echo "targets=${TARGETS[*]}"
  echo "gpus=${GPUS[*]}"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$LOG_ROOT/launch/provenance.txt"

pids=()
for index in "${!GPUS[@]}"; do
  worker "$index" "${GPUS[$index]}" & pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || exit "$status"

observed="$(find "$RESULT_ROOT" -maxdepth 1 -type f -name '*.json' | wc -l | tr -d ' ')"
[[ "$observed" -eq 20 ]] || { echo "expected 20 result files, found $observed" >&2; exit 1; }
{
  echo "completed_utc=$(date -u +%FT%TZ)"
  echo "physical_cells=$observed"
} > "$LOG_ROOT/COMPLETE"
cat "$LOG_ROOT/COMPLETE"
