#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/dataMeR1/phil/gfm/worktree-runtime-archive-20260812}"
ARCH100_STATE="${ARCH100_STATE:-${ARCHIVE_ROOT}/prodigy-archmatrix/files/state/icl_arch_matrix}"
FINAL2500_STATE="${FINAL2500_STATE:-${ARCHIVE_ROOT}/prodigy-final-core/files/state/final_core}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/ladder_cross_task_eval}"
STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state/ladder_cross_task_eval}"
NM_REFERENCE="${NM_REFERENCE:-${REPO_ROOT}/scripts/experiments/analysis/transfer/matrices/cross_model/final_core/data/prodigy_final_core/fixed_test/summary/episode_fingerprints.tsv}"
DOWNSTREAM_REFERENCE="${DOWNSTREAM_REFERENCE:-${REPO_ROOT}/scripts/experiments/analysis/transfer/matrices/cross_architecture/icl_arch_matrix/data/raw_aggregate/prodigy.jsonl}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
MAX_EXISTING_GPU_MIB="${MAX_EXISTING_GPU_MIB:-12000}"
MIN_HOST_AVAILABLE_GIB="${MIN_HOST_AVAILABLE_GIB:-700}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
export FINAL_CORE_CPU_THREADS="${FINAL_CORE_CPU_THREADS:-8}"
export OMP_NUM_THREADS="$FINAL_CORE_CPU_THREADS"
export MKL_NUM_THREADS="$FINAL_CORE_CPU_THREADS"
export OPENBLAS_NUM_THREADS="$FINAL_CORE_CPU_THREADS"
PYTHON="${CONDA_PREFIX}/bin/python"

READY_DIR="$LOG_ROOT/nm100/ready/$RUN_ID"
mkdir -p "$LOG_ROOT" "$STATE_ROOT" "$LOG_ROOT/nm100/results" \
  "$READY_DIR" "$LOG_ROOT/downstream2500" "$LOG_ROOT/summary"
cd "$REPO_ROOT"

for gpu in 0 1; do
  used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if (( used > MAX_EXISTING_GPU_MIB )); then
    echo "REFUSE gpu=$gpu already uses ${used}MiB; limit=${MAX_EXISTING_GPU_MIB}MiB" >&2
    exit 2
  fi
done
available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
available_gib=$((available_kib / 1024 / 1024))
if (( available_gib < MIN_HOST_AVAILABLE_GIB )); then
  echo "REFUSE host available=${available_gib}GiB; need ${MIN_HOST_AVAILABLE_GIB}GiB" >&2
  exit 2
fi

echo "START nm100 run_id=$RUN_ID host_available_gib=$available_gib"
pids=()
for worker in 0 1; do
  CUDA_VISIBLE_DEVICES="$worker" "$PYTHON" -u -m \
    scripts.experiments.setup.ladder_cross_task_eval.evaluate_nm100 \
    --worker-index "$worker" --worker-count 2 --expected-workers 2 \
    --targets "ukr_rus,covid,midterm,covid_political,election2020,ukr_rus_suspended,twibot20,cp_hk,facebook_page_reference" \
    --batch-size 32 --episode-count 512 \
    --config scripts/experiments/setup/final_core/training.yaml \
    --training-state-root "$ARCH100_STATE" --training-run-stamp 20260810 \
    --evaluation-state-root "$STATE_ROOT/nm100/worker${worker}" \
    --evaluation-log-root "$LOG_ROOT/nm100/internal/worker${worker}" \
    --results-root "$LOG_ROOT/nm100/results" \
    --evaluation-run-stamp "${RUN_ID}_nm100_w${worker}" \
    --ready-dir "$READY_DIR" --min-host-reserve-gib 400 \
    --reference-fingerprints "$NM_REFERENCE" \
    > "$LOG_ROOT/nm100_worker${worker}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || { echo "NM100_FAILED" >&2; exit 1; }
echo "DONE nm100"

echo "START downstream2500"
pids=()
for worker in 0 1; do
  CUDA_VISIBLE_DEVICES="$worker" "$PYTHON" -u -m \
    scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy \
    --config scripts/experiments/setup/final_core/training.yaml \
    --state-root "$FINAL2500_STATE" --checkpoint-layout final-core \
    --checkpoint-step 2500 --training-seeds 0,1,2 --ladder-only \
    --worker-index "$worker" --worker-count 2 \
    --eval-state-root "$STATE_ROOT/downstream2500/worker${worker}" \
    --log-root "$LOG_ROOT/downstream2500/internal/worker${worker}" \
    --results "$LOG_ROOT/downstream2500/worker${worker}.jsonl" \
    --run-stamp 20260807 --device 0 --resume \
    --reference-results "$DOWNSTREAM_REFERENCE" \
    > "$LOG_ROOT/downstream2500_worker${worker}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || { echo "DOWNSTREAM2500_FAILED" >&2; exit 1; }
echo "DONE downstream2500"

"$PYTHON" -m scripts.experiments.setup.ladder_cross_task_eval.aggregate \
  --nm-results-root "$LOG_ROOT/nm100/results" \
  --downstream-worker-results "$LOG_ROOT/downstream2500/worker0.jsonl" \
    "$LOG_ROOT/downstream2500/worker1.jsonl" \
  --output-root "$LOG_ROOT/summary"
date -u +%FT%TZ > "$LOG_ROOT/complete_utc.txt"
echo "LADDER_CROSS_TASK_EVAL_COMPLETE summary=$LOG_ROOT/summary"
