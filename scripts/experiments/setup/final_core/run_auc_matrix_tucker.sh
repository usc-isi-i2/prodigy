#!/usr/bin/env bash
# Three-seed 9x9 specialist matrix with preserved multiclass ROC-AUC.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TRAINING_STATE_ROOT="${TRAINING_STATE_ROOT:-/dataMeR1/phil/gfm/prodigy-final-core/state/final_core}"
TRAINING_RUN_STAMP="${TRAINING_RUN_STAMP:-20260807}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
EVAL_STATE_ROOT="${EVAL_STATE_ROOT:-${REPO_ROOT}/state/final_core_auc}"
EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${REPO_ROOT}/log/final_core_auc}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKER_COUNT=8
MIN_HOST_RESERVE_GIB="${MIN_HOST_RESERVE_GIB:-256}"
PRELOAD_GIB_PER_WORKER="${PRELOAD_GIB_PER_WORKER:-125}"
CPU_THREADS_PER_WORKER="${CPU_THREADS_PER_WORKER:-24}"
REFERENCE_FINGERPRINTS="${REFERENCE_FINGERPRINTS:-/dataMeR1/phil/gfm/prodigy-final-core-cache/log/final_core_cached_test/production/bs32/summary/episode_fingerprints.tsv}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
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
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${ACTIVE_PIDS[@]:-}"; do
    [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
  done
  ACTIVE_PIDS=()
}
trap cleanup_workers EXIT INT TERM

mkdir -p "$EVAL_STATE_ROOT" "$EVAL_LOG_ROOT/queue" "$EVAL_LOG_ROOT/ready"
cd "$REPO_ROOT"

missing=0
for seed in 0 1 2; do
  for source in ukr_rus covid midterm covid_political election2020 ukr_rus_suspended twibot20 cp_hk facebook_page_reference; do
    checkpoint="$TRAINING_STATE_ROOT/finalcore_ss_${source}_s${seed}_${TRAINING_RUN_STAMP}/checkpoint/state_dict_2500.ckpt"
    if [[ ! -f "$checkpoint" ]]; then
      echo "MISSING $checkpoint" >&2
      missing=$((missing + 1))
    fi
  done
done
(( missing == 0 )) || { echo "$missing specialist checkpoints are missing" >&2; exit 1; }
[[ -f "$REFERENCE_FINGERPRINTS" ]] || {
  echo "MISSING reference fingerprint ledger $REFERENCE_FINGERPRINTS" >&2
  exit 1
}

wait_for_resources() {
  local required_gib=$((WORKER_COUNT * PRELOAD_GIB_PER_WORKER + MIN_HOST_RESERVE_GIB))
  while true; do
    local clear=1 gpu used available_kib available_gib
    for gpu in 0 1 2 3; do
      used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
      if (( used > 1000 )); then
        echo "WAIT GPU $gpu is busy (${used} MiB); not launching or stopping it" >&2
        clear=0
      fi
    done
    available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    available_gib=$((available_kib / 1024 / 1024))
    if (( available_gib < required_gib )); then
      echo "WAIT host RAM ${available_gib} GiB; need ${required_gib} GiB" >&2
      clear=0
    fi
    if (( clear == 1 )); then
      echo "RESOURCE_GATE_OK available_host_gib=$available_gib utc=$(date -u +%FT%TZ)"
      return 0
    fi
    sleep 30
  done
}

smoke_targets() {
  case "$1" in
    0) echo "ukr_rus,facebook_page_reference" ;;
    1) echo "ukr_rus,cp_hk" ;;
    2) echo "covid" ;;
    3) echo "midterm" ;;
    4) echo "covid_political" ;;
    5) echo "election2020" ;;
    6) echo "ukr_rus_suspended" ;;
    7) echo "twibot20" ;;
  esac
}

launch_workers() {
  local kind="$1" results_root="$2" ready_dir="$3"
  local status=0 worker gpu targets
  local -a pids=()
  mkdir -p "$results_root" "$ready_dir"
  find "$ready_dir" -maxdepth 1 -type f -name 'worker_*.json' -delete
  for worker in 0 1 2 3 4 5 6 7; do
    gpu=$((worker / 2))
    targets="ukr_rus,covid,midterm,covid_political,election2020,ukr_rus_suspended,twibot20,cp_hk,facebook_page_reference"
    [[ "$kind" == smoke ]] && targets="$(smoke_targets "$worker")"
    cmd=("$PYTHON" -u "$SCRIPT_DIR/evaluate_fixed_grid.py"
         --specialists-only --worker-index "$worker" --worker-count "$WORKER_COUNT"
         --targets "$targets" --batch-size "$BATCH_SIZE" --episode-count 512
         --config "$SCRIPT_DIR/training.yaml"
         --training-state-root "$TRAINING_STATE_ROOT"
         --training-run-stamp "$TRAINING_RUN_STAMP"
         --evaluation-state-root "$EVAL_STATE_ROOT/${kind}_bs${BATCH_SIZE}"
         --evaluation-log-root "$EVAL_LOG_ROOT/internal/${kind}_bs${BATCH_SIZE}"
         --results-root "$results_root"
         --evaluation-run-stamp "${RUN_ID}_${kind}_bs${BATCH_SIZE}"
         --reference-fingerprints "$REFERENCE_FINGERPRINTS"
         --ready-dir "$ready_dir" --expected-workers "$WORKER_COUNT"
         --min-host-reserve-gib "$MIN_HOST_RESERVE_GIB")
    [[ "$kind" == smoke ]] && cmd+=(--max-checkpoints 1)
    echo "LAUNCH kind=$kind worker=$worker gpu=$gpu targets=$targets"
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" \
      > "$EVAL_LOG_ROOT/queue/${kind}_worker${worker}.log" 2>&1 &
    pids+=("$!")
  done
  ACTIVE_PIDS=("${pids[@]}")
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  ACTIVE_PIDS=()
  return "$status"
}

if [[ "$SKIP_SMOKE" != 1 ]]; then
  smoke_root="$EVAL_LOG_ROOT/smoke/bs${BATCH_SIZE}/${RUN_ID}/results"
  smoke_ready="$EVAL_LOG_ROOT/ready/smoke_${RUN_ID}"
  wait_for_resources
  launch_workers smoke "$smoke_root" "$smoke_ready"
  "$PYTHON" "$SCRIPT_DIR/verify_fixed_smoke.py" \
    --results-root "$smoke_root" --ready-dir "$smoke_ready" \
    --expected-workers 8 --expected-cells 10 --batch-size "$BATCH_SIZE" \
    --max-summed-vram-gib-per-gpu 70 \
    --min-host-reserve-gib "$MIN_HOST_RESERVE_GIB" \
    > "$EVAL_LOG_ROOT/smoke/bs${BATCH_SIZE}/${RUN_ID}/verification.json"
fi
[[ "$SMOKE_ONLY" == 1 ]] && { echo "FINAL_CORE_AUC_SMOKE_COMPLETE"; exit 0; }

results_root="$EVAL_LOG_ROOT/production/bs${BATCH_SIZE}/results"
summary_root="$EVAL_LOG_ROOT/production/bs${BATCH_SIZE}/summary"
ready_dir="$EVAL_LOG_ROOT/ready/production_${RUN_ID}"
mkdir -p "$results_root" "$summary_root" "$ready_dir"
{
  echo "protocol=fixed_test_512_static_test_on_static_train_v1"
  echo "metric_contract=accuracy_f1_macro_roc_auc_ovr_macro_v1"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "training_state_root=$TRAINING_STATE_ROOT"
  echo "checkpoint_step=2500"
  echo "specialist_cells=243"
  echo "reference_fingerprints=$REFERENCE_FINGERPRINTS"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$EVAL_LOG_ROOT/production/bs${BATCH_SIZE}/provenance.txt"

wait_for_resources
launch_workers production "$results_root" "$ready_dir"
"$PYTHON" "$SCRIPT_DIR/aggregate_auc_matrix.py" \
  --results-root "$results_root" --output-root "$summary_root" \
  --expected-batch-size "$BATCH_SIZE"
date -u +%FT%TZ > "$EVAL_LOG_ROOT/production/bs${BATCH_SIZE}/complete_utc.txt"
echo "FINAL_CORE_AUC_COMPLETE summary=$summary_root"
