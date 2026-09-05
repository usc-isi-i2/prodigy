#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TRAINING_RUN_DIR="${TRAINING_RUN_DIR:-${REPO_ROOT}/log/nm_leave_one_out_finalcore/shared_seed0_20260904}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
EVAL_STATE_ROOT="${EVAL_STATE_ROOT:-${REPO_ROOT}/state/nm_leave_one_out_finalcore_eval}"
EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${REPO_ROOT}/log/nm_leave_one_out_finalcore_eval}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKER_COUNT=9
FIRST_WAVE_WORKERS=8
MIN_HOST_RESERVE_GIB="${MIN_HOST_RESERVE_GIB:-256}"
PRELOAD_GIB_PER_WORKER="${PRELOAD_GIB_PER_WORKER:-125}"
MAX_EXISTING_GPU_MIB="${MAX_EXISTING_GPU_MIB:-1000}"
CPU_THREADS_PER_WORKER="${CPU_THREADS_PER_WORKER:-24}"
REFERENCE_FINGERPRINTS="${REFERENCE_FINGERPRINTS:-${REPO_ROOT}/scripts/experiments/analysis/transfer/matrices/cross_model/final_core/data/prodigy_final_core/auc/reference/episode_fingerprints.tsv}"
DRY_RUN="${DRY_RUN:-0}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export FINAL_CORE_CPU_THREADS="$CPU_THREADS_PER_WORKER"
export OMP_NUM_THREADS="$CPU_THREADS_PER_WORKER"
export MKL_NUM_THREADS="$CPU_THREADS_PER_WORKER"
export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_WORKER"
export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_WORKER"
export NUMEXPR_MAX_THREADS="$CPU_THREADS_PER_WORKER"
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"
ACTIVE_PIDS=()

cleanup_workers() {
  local pid
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid" 2>/dev/null || true; fi
  done
  for pid in "${ACTIVE_PIDS[@]:-}"; do [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true; done
  ACTIVE_PIDS=()
}
trap cleanup_workers EXIT INT TERM

mkdir -p "$EVAL_STATE_ROOT" "$EVAL_LOG_ROOT/queue" "$EVAL_LOG_ROOT/ready"
cd "$REPO_ROOT"
[[ -f "$REFERENCE_FINGERPRINTS" ]] || { echo "missing fingerprint ledger $REFERENCE_FINGERPRINTS" >&2; exit 1; }
if [[ "$DRY_RUN" != 1 ]]; then
  "$PYTHON" "$SCRIPT_DIR/verify_training.py" --run-dir "$TRAINING_RUN_DIR" \
    > "$EVAL_LOG_ROOT/training_verification.json"
fi

resource_gate() {
  local worker_total="$1"
  local required_gib=$((worker_total * PRELOAD_GIB_PER_WORKER + MIN_HOST_RESERVE_GIB))
  local gpu used available_kib available_gib
  for gpu in 0 1 2 3; do
    used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    (( used <= MAX_EXISTING_GPU_MIB )) || {
      echo "GPU $gpu is busy (${used} MiB > ${MAX_EXISTING_GPU_MIB} MiB); refusing launch" >&2
      return 1
    }
  done
  available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  available_gib=$((available_kib / 1024 / 1024))
  (( available_gib >= required_gib )) || {
    echo "need ${required_gib} GiB host RAM for ${worker_total} graph loads; only ${available_gib} GiB available" >&2
    return 1
  }
}

target_for_worker() {
  case "$1" in
    0) echo "ukr_rus" ;;
    1) echo "covid" ;;
    2) echo "midterm" ;;
    3) echo "covid_political" ;;
    4) echo "election2020" ;;
    5) echo "ukr_rus_suspended" ;;
    6) echo "twibot20" ;;
    7) echo "cp_hk" ;;
    8) echo "facebook_page_reference" ;;
    *) return 2 ;;
  esac
}

launch_wave() {
  local wave="$1" first="$2" last="$3" expected="$4" ready_dir="$5"
  local status=0 worker gpu target
  local -a pids=()
  mkdir -p "$ready_dir"
  find "$ready_dir" -maxdepth 1 -type f -name 'worker_*.json' -delete
  for ((worker=first; worker<=last; worker++)); do
    gpu=$((worker / 2))
    (( gpu > 3 )) && gpu=0
    target="$(target_for_worker "$worker")"
    cmd=("$PYTHON" -u "$SCRIPT_DIR/evaluate_loo.py"
         --worker-index "$worker" --worker-count "$WORKER_COUNT"
         --targets "$target" --batch-size "$BATCH_SIZE" --episode-count 512
         --config "$SCRIPT_DIR/training.yaml"
         --training-state-root "$TRAINING_RUN_DIR" --training-run-stamp ignored
         --evaluation-state-root "$EVAL_STATE_ROOT/production_bs${BATCH_SIZE}"
         --evaluation-log-root "$EVAL_LOG_ROOT/internal/production_bs${BATCH_SIZE}"
         --results-root "$RESULTS_ROOT"
         --evaluation-run-stamp "${RUN_ID}_${wave}_bs${BATCH_SIZE}"
         --reference-fingerprints "$REFERENCE_FINGERPRINTS"
         --ready-dir "$ready_dir" --expected-workers "$expected"
         --min-host-reserve-gib "$MIN_HOST_RESERVE_GIB")
    if [[ "$DRY_RUN" == 1 ]]; then
      printf 'DRY wave=%s worker=%s gpu=%s target=%s' "$wave" "$worker" "$gpu" "$target"
      printf ' %q' "${cmd[@]}"; printf '\n'
    else
      CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" \
        > "$EVAL_LOG_ROOT/queue/${wave}_worker${worker}.log" 2>&1 &
      pids+=("$!")
    fi
  done
  [[ "$DRY_RUN" == 1 ]] && return 0
  ACTIVE_PIDS=("${pids[@]}")
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  ACTIVE_PIDS=()
  return "$status"
}

RESULTS_ROOT="$EVAL_LOG_ROOT/production/bs${BATCH_SIZE}/results"
SUMMARY_ROOT="$EVAL_LOG_ROOT/production/bs${BATCH_SIZE}/summary"
mkdir -p "$RESULTS_ROOT" "$SUMMARY_ROOT"
if [[ "$DRY_RUN" != 1 ]]; then
  {
    echo "protocol=fixed_test_512_static_test_on_static_train_v1"
    echo "metric_contract=accuracy_f1_macro_roc_auc_ovr_macro_v1"
    echo "commit=$(git rev-parse HEAD)"
    echo "branch=$(git rev-parse --abbrev-ref HEAD)"
    echo "training_run_dir=$TRAINING_RUN_DIR"
    echo "checkpoint_step=2500"
    echo "loo_models=9"
    echo "heldout_test_cells=9"
    echo "started_utc=$(date -u +%FT%TZ)"
  } > "$EVAL_LOG_ROOT/production/bs${BATCH_SIZE}/provenance.txt"
  resource_gate "$FIRST_WAVE_WORKERS"
fi
launch_wave wave1 0 7 8 "$EVAL_LOG_ROOT/ready/wave1_${RUN_ID}"
if [[ "$DRY_RUN" != 1 ]]; then resource_gate 1; fi
launch_wave wave2 8 8 1 "$EVAL_LOG_ROOT/ready/wave2_${RUN_ID}"
[[ "$DRY_RUN" == 1 ]] && exit 0
"$PYTHON" "$SCRIPT_DIR/aggregate_results.py" \
  --results-root "$RESULTS_ROOT" --output-root "$SUMMARY_ROOT" \
  --training-run-dir "$TRAINING_RUN_DIR" --expected-batch-size "$BATCH_SIZE"
date -u +%FT%TZ > "$EVAL_LOG_ROOT/production/bs${BATCH_SIZE}/complete_utc.txt"
