#!/usr/bin/env bash
# Replay only stage-1 specialists on the fixed Facebook discovery stream.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_STAMP="${RUN_STAMP:-20260907v1}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/trace_health_guided}"
MODEL_LIST="${MODEL_LIST:-${LOG_ROOT}/launch/stage1_model_list_${RUN_STAMP}.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${LOG_ROOT}/health_replay_${RUN_STAMP}}"
GPU="${GPU:-0}"
TRACE_PARITY_ATOL="${TRACE_PARITY_ATOL:-1e-5}"

[[ "$GPU" =~ ^[0-3]$ ]] || { echo "unauthorized GPU $GPU" >&2; exit 2; }
[[ -f "$MODEL_LIST" ]] || { echo "missing model list: $MODEL_LIST" >&2; exit 2; }

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"

job_root="$OUTPUT_ROOT/original/facebook_page_reference"
directory="$job_root/facebook_page_reference"
log="$LOG_ROOT/replay_logs/health_original_facebook_${RUN_STAMP}.log"
mkdir -p "$LOG_ROOT/replay_logs"
if [[ -f "$job_root/DONE" ]]; then
  echo "SKIP complete specialist health replay: $job_root"
elif [[ -e "$job_root" ]]; then
  echo "REFUSE incomplete specialist health replay: $job_root" >&2
  exit 1
else
  mkdir -p "$(dirname "$job_root")"
  "$PYTHON" -u -m scripts.experiments.setup.target_performance_mechanisms.replay \
    --model-list "$MODEL_LIST" --output "$job_root" \
    --datasets facebook_page_reference --variants baseline \
    --specialists-only --device "$GPU" --threads 4 --batch-count 32 \
    --save-embeddings --training-label-count 30 \
    --trace-parity-atol "$TRACE_PARITY_ATOL" \
    --eval-episode-seed-offset 0 > "$log" 2>&1
fi
[[ -f "$job_root/DONE" && -d "$directory" ]] || {
  echo "incomplete specialist health replay: $job_root" >&2
  exit 1
}
echo "TRACE_HEALTH_SPECIALIST_REPLAY_COMPLETE directory=$directory"
