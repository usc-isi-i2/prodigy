#!/usr/bin/env bash
# Replay all terminal schedule models on disjoint original/fresh target streams.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_STAMP="${RUN_STAMP:-20260906v1}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/trace_schedule_scaling}"
MODEL_LIST="${MODEL_LIST:-${LOG_ROOT}/launch/model_list_${RUN_STAMP}.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${LOG_ROOT}/replay_${RUN_STAMP}}"
GPUS_TEXT="${GPUS:-0 1 2 3}"
TARGETS_TEXT="${TARGETS:-election2020 ukr_rus_suspended twibot20 cp_hk facebook_page_reference}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONDONTWRITEBYTECODE=1
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"

read -r -a GPU_IDS <<< "$GPUS_TEXT"
read -r -a TARGETS <<< "$TARGETS_TEXT"
[[ -f "$MODEL_LIST" ]] || { echo "missing model list: $MODEL_LIST" >&2; exit 2; }
for gpu in "${GPU_IDS[@]}"; do
  [[ "$gpu" =~ ^[0-3]$ ]] || { echo "unauthorized GPU $gpu" >&2; exit 2; }
done

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT/replay_logs"
jobs=()
for stream in original fresh; do
  offset=0
  [[ "$stream" == fresh ]] && offset=100003
  for target in "${TARGETS[@]}"; do
    jobs+=("$stream|$offset|$target")
  done
done

worker() {
  local worker_index="$1" gpu="$2" index=0 item stream offset target output log
  for item in "${jobs[@]}"; do
    if (( index % ${#GPU_IDS[@]} == worker_index )); then
      IFS='|' read -r stream offset target <<< "$item"
      output="$OUTPUT_ROOT/$stream/$target"
      log="$LOG_ROOT/replay_logs/${stream}_${target}_${RUN_STAMP}.log"
      if [[ -f "$output/DONE" ]]; then
        echo "[gpu $gpu] SKIP replay $stream/$target"
      elif [[ -e "$output" ]]; then
        echo "[gpu $gpu] REFUSE incomplete replay $output" >&2
        return 1
      else
        echo "[gpu $gpu] START replay $stream/$target utc=$(date -u +%FT%TZ)"
        "$PYTHON" -u -m scripts.experiments.setup.target_performance_mechanisms.replay \
          --model-list "$MODEL_LIST" --output "$output" --datasets "$target" \
          --variants baseline --device "$gpu" --threads 4 --batch-count 32 \
          --save-embeddings --training-label-count 30 \
          --eval-episode-seed-offset "$offset" > "$log" 2>&1
        [[ -f "$output/DONE" ]] || { echo "incomplete replay $output" >&2; return 1; }
        echo "[gpu $gpu] DONE replay $stream/$target utc=$(date -u +%FT%TZ)"
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
echo "TRACE_SCHEDULE_REPLAY_COMPLETE output=$OUTPUT_ROOT"
