#!/usr/bin/env bash
# Combined 837-cell fixed-test sweep. No validation and no checkpoint selection.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TRAINING_STATE_ROOT="${TRAINING_STATE_ROOT:-/dataMeR1/phil/gfm/prodigy-final-core/state/final_core}"
TRAINING_RUN_STAMP="${TRAINING_RUN_STAMP:-20260807}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
EVAL_STATE_ROOT="${EVAL_STATE_ROOT:-${REPO_ROOT}/state/final_core_fixed_test}"
EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${REPO_ROOT}/log/final_core_fixed_test}"
PREFERRED_BATCH_SIZE="${PREFERRED_BATCH_SIZE:-32}"
FALLBACK_BATCH_SIZE="${FALLBACK_BATCH_SIZE:-32}"
WORKER_COUNT=8
MIN_HOST_RESERVE_GIB="${MIN_HOST_RESERVE_GIB:-256}"
MAX_SUMMED_VRAM_GIB="${MAX_SUMMED_VRAM_GIB:-70}"
PRELOAD_GIB_PER_WORKER="${PRELOAD_GIB_PER_WORKER:-125}"
CPU_THREADS_PER_WORKER="${CPU_THREADS_PER_WORKER:-24}"
SMOKE_MAX_CHECKPOINTS="${SMOKE_MAX_CHECKPOINTS:-2}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
PRODUCTION_RESULTS_ROOT="${PRODUCTION_RESULTS_ROOT:-}"

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
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
  ACTIVE_PIDS=()
}
trap cleanup_workers EXIT INT TERM

mkdir -p "$EVAL_STATE_ROOT" "$EVAL_LOG_ROOT" "$EVAL_LOG_ROOT/queue" "$EVAL_LOG_ROOT/ready"
cd "$REPO_ROOT"

PLAN="$EVAL_LOG_ROOT/physical_plan.tsv"
"$PYTHON" "$SCRIPT_DIR/fixed_test_plan.py" > "$PLAN"
[[ "$(($(wc -l < "$PLAN") - 1))" == 93 ]] || {
  echo "physical plan must contain 93 checkpoint-seed jobs" >&2
  exit 2
}

missing=0
while IFS=$'\t' read -r _index seed model_id _n_sources _sources _aliases; do
  [[ "$seed" == seed ]] && continue
  checkpoint="$TRAINING_STATE_ROOT/finalcore_${model_id}_s${seed}_${TRAINING_RUN_STAMP}/checkpoint/state_dict_2500.ckpt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "MISSING $checkpoint" >&2
    missing=$((missing + 1))
  fi
done < "$PLAN"
(( missing == 0 )) || { echo "$missing step-2500 checkpoints are missing" >&2; exit 1; }

wait_for_resources() {
  local required_gib=$((WORKER_COUNT * PRELOAD_GIB_PER_WORKER + MIN_HOST_RESERVE_GIB))
  while true; do
    local clear=1 gpu used available_kib available_gib
    for gpu in 0 1 2 3; do
      used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
      if (( used > 1000 )); then
        echo "WAIT GPU $gpu is busy (${used} MiB); not launching or stopping it" >&2
        clear=0
      elif ! CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -c \
          'import torch; x=torch.ones(1, device="cuda"); assert int(x.item()) == 1' \
          >/dev/null 2>&1; then
        echo "WAIT GPU $gpu is visible to nvidia-smi but fails a CUDA health probe; " \
             "an administrator reset may be required" >&2
        clear=0
      fi
    done
    available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
    available_gib=$((available_kib / 1024 / 1024))
    if (( available_gib < required_gib )); then
      echo "WAIT host RAM ${available_gib} GiB; need ${required_gib} GiB before eight graph loads" >&2
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
    *) return 2 ;;
  esac
}

launch_workers() {
  local kind="$1" batch_size="$2" results_root="$3" ready_dir="$4"
  local status=0 worker gpu targets
  local -a pids=()
  mkdir -p "$results_root" "$ready_dir"
  find "$ready_dir" -maxdepth 1 -type f -name 'worker_*.json' -delete
  for worker in 0 1 2 3 4 5 6 7; do
    gpu=$((worker / 2))
    if [[ "$kind" == smoke ]]; then
      targets="$(smoke_targets "$worker")"
    else
      targets="ukr_rus,covid,midterm,covid_political,election2020,ukr_rus_suspended,twibot20,cp_hk,facebook_page_reference"
    fi
    cmd=("$PYTHON" -u "$SCRIPT_DIR/evaluate_fixed_grid.py"
         --worker-index "$worker" --worker-count "$WORKER_COUNT"
         --targets "$targets" --batch-size "$batch_size" --episode-count 512
         --config "$SCRIPT_DIR/training.yaml"
         --training-state-root "$TRAINING_STATE_ROOT"
         --training-run-stamp "$TRAINING_RUN_STAMP"
         --evaluation-state-root "$EVAL_STATE_ROOT/${kind}_bs${batch_size}"
         --evaluation-log-root "$EVAL_LOG_ROOT/internal/${kind}_bs${batch_size}"
         --results-root "$results_root"
         --evaluation-run-stamp "${RUN_ID}_${kind}_bs${batch_size}"
         --ready-dir "$ready_dir" --expected-workers "$WORKER_COUNT"
         --min-host-reserve-gib "$MIN_HOST_RESERVE_GIB")
    if [[ "$kind" == smoke ]]; then
      cmd+=(--max-checkpoints "$SMOKE_MAX_CHECKPOINTS")
    fi
    echo "LAUNCH kind=$kind batch=$batch_size worker=$worker gpu=$gpu targets=$targets"
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" \
      > "$EVAL_LOG_ROOT/queue/${kind}_bs${batch_size}_worker${worker}.log" 2>&1 &
    pids+=("$!")
  done
  ACTIVE_PIDS=("${pids[@]}")
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  ACTIVE_PIDS=()
  return "$status"
}

run_smoke() {
  local batch_size="$1"
  local smoke_root="$EVAL_LOG_ROOT/smoke/bs${batch_size}/${RUN_ID}/results"
  local ready_dir="$EVAL_LOG_ROOT/ready/smoke_bs${batch_size}_${RUN_ID}"
  local expected_cells=$((10 * SMOKE_MAX_CHECKPOINTS))
  wait_for_resources
  if ! launch_workers smoke "$batch_size" "$smoke_root" "$ready_dir"; then
    echo "SMOKE_WORKER_FAILURE batch_size=$batch_size" >&2
    return 1
  fi
  "$PYTHON" "$SCRIPT_DIR/verify_fixed_smoke.py" \
    --results-root "$smoke_root" --ready-dir "$ready_dir" \
    --expected-workers "$WORKER_COUNT" --expected-cells "$expected_cells" \
    --batch-size "$batch_size" \
    --max-summed-vram-gib-per-gpu "$MAX_SUMMED_VRAM_GIB" \
    --min-host-reserve-gib "$MIN_HOST_RESERVE_GIB" \
    > "$EVAL_LOG_ROOT/smoke/bs${batch_size}/${RUN_ID}/verification.json"
}

selected_batch_size="$PREFERRED_BATCH_SIZE"
if [[ "$SKIP_SMOKE" == 1 ]]; then
  echo "SMOKE_SKIPPED by explicit SKIP_SMOKE=1"
elif run_smoke "$PREFERRED_BATCH_SIZE"; then
  echo "SMOKE_OK using preferred batch size $PREFERRED_BATCH_SIZE"
else
  if [[ "$FALLBACK_BATCH_SIZE" == "$PREFERRED_BATCH_SIZE" ]]; then
    echo "Smoke failed at the configured batch size $PREFERRED_BATCH_SIZE" >&2
    exit 1
  fi
  echo "Preferred batch size $PREFERRED_BATCH_SIZE was unsafe; trying $FALLBACK_BATCH_SIZE" >&2
  selected_batch_size="$FALLBACK_BATCH_SIZE"
  run_smoke "$FALLBACK_BATCH_SIZE"
  echo "SMOKE_OK using fallback batch size $FALLBACK_BATCH_SIZE"
fi

if [[ "$SMOKE_ONLY" == 1 ]]; then
  echo "FINAL_CORE_CACHED_SMOKE_COMPLETE batch_size=$selected_batch_size"
  exit 0
fi

RESULTS_ROOT="${PRODUCTION_RESULTS_ROOT:-$EVAL_LOG_ROOT/production/bs${selected_batch_size}/results}"
SUMMARY_ROOT="$EVAL_LOG_ROOT/production/bs${selected_batch_size}/summary"
READY_DIR="$EVAL_LOG_ROOT/ready/production_bs${selected_batch_size}_${RUN_ID}"
mkdir -p "$RESULTS_ROOT" "$SUMMARY_ROOT" "$READY_DIR"

{
  echo "protocol=fixed_test_512_static_test_on_static_train_v1"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "training_state_root=$TRAINING_STATE_ROOT"
  echo "training_run_stamp=$TRAINING_RUN_STAMP"
  echo "checkpoint_step=2500"
  echo "batch_size=$selected_batch_size"
  echo "batch_count=$((512 / selected_batch_size))"
  echo "episode_count=512"
  echo "batch_replay_mode=materialized_cpu_clone_v1"
  echo "workers=8"
  echo "gpus=0,1,2,3"
  echo "slots_per_gpu=2"
  echo "cpu_threads_per_worker=$CPU_THREADS_PER_WORKER"
  echo "production_results_root=$RESULTS_ROOT"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$EVAL_LOG_ROOT/production/bs${selected_batch_size}/provenance.txt"

wait_for_resources
launch_workers production "$selected_batch_size" "$RESULTS_ROOT" "$READY_DIR"
"$PYTHON" "$SCRIPT_DIR/aggregate_fixed_test.py" \
  --results-root "$RESULTS_ROOT" --output-root "$SUMMARY_ROOT" \
  --expected-batch-size "$selected_batch_size"
date -u +%FT%TZ > "$EVAL_LOG_ROOT/production/bs${selected_batch_size}/complete_utc.txt"
echo "FINAL_CORE_FIXED_TEST_COMPLETE batch_size=$selected_batch_size summary=$SUMMARY_ROOT"
