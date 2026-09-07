#!/usr/bin/env bash
# Evaluate the complete matched-budget method lattice on original and fresh streams.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_STAMP="${RUN_STAMP:-20260907v1}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/trace_health_guided}"
MODEL_LIST="${MODEL_LIST:-${LOG_ROOT}/launch/final_model_list_${RUN_STAMP}.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${LOG_ROOT}/final_replay_${RUN_STAMP}}"
GPUS_TEXT="${GPUS:-0 1}"
TRACE_PARITY_ATOL="${TRACE_PARITY_ATOL:-1e-5}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"

read -r -a GPU_IDS <<< "$GPUS_TEXT"
[[ -f "$MODEL_LIST" ]] || { echo "missing model list: $MODEL_LIST" >&2; exit 2; }
for gpu in "${GPU_IDS[@]}"; do
  [[ "$gpu" =~ ^[0-3]$ ]] || { echo "unauthorized GPU $gpu" >&2; exit 2; }
done

jobs=("original|0" "fresh|100003")
worker() {
  local worker_index="$1" gpu="$2" index=0 item stream offset job_root log
  for item in "${jobs[@]}"; do
    if (( index % ${#GPU_IDS[@]} == worker_index )); then
      IFS='|' read -r stream offset <<< "$item"
      job_root="$OUTPUT_ROOT/$stream/facebook_page_reference"
      log="$LOG_ROOT/replay_logs/final_${stream}_facebook_${RUN_STAMP}.log"
      if [[ -f "$job_root/DONE" ]]; then
        echo "[gpu $gpu] SKIP final replay $stream"
      elif [[ -e "$job_root" ]]; then
        echo "[gpu $gpu] REFUSE incomplete replay $job_root" >&2
        return 1
      else
        mkdir -p "$(dirname "$job_root")" "$LOG_ROOT/replay_logs"
        echo "[gpu $gpu] START final replay $stream utc=$(date -u +%FT%TZ)"
        "$PYTHON" -u -m scripts.experiments.setup.target_performance_mechanisms.replay \
          --model-list "$MODEL_LIST" --output "$job_root" \
          --datasets facebook_page_reference --variants baseline \
          --device "$gpu" --threads 4 --batch-count 32 \
          --training-label-count 30 --trace-parity-atol "$TRACE_PARITY_ATOL" \
          --eval-episode-seed-offset "$offset" > "$log" 2>&1
        [[ -f "$job_root/DONE" ]] || { echo "incomplete replay $job_root" >&2; return 1; }
        echo "[gpu $gpu] DONE final replay $stream utc=$(date -u +%FT%TZ)"
      fi
    fi
    ((index+=1))
  done
}

pids=()
for index in "${!GPU_IDS[@]}"; do
  worker "$index" "${GPU_IDS[$index]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || exit "$status"
echo "TRACE_HEALTH_FINAL_REPLAY_COMPLETE output=$OUTPUT_ROOT"
