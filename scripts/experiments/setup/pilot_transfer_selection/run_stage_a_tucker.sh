#!/usr/bin/env bash
# Validation-only early-checkpoint transfer matrix. The user launches production.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
FINAL_CORE_DIR="${REPO_ROOT}/scripts/experiments/setup/final_core"
TRAINING_STATE_ROOT="${TRAINING_STATE_ROOT:-/dataMeR1/phil/gfm/worktree-runtime-archive-20260812/prodigy-final-core/files/state/final_core}"
EVAL_STATE_ROOT="${EVAL_STATE_ROOT:-${REPO_ROOT}/state/pilot_transfer_selection}"
EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${REPO_ROOT}/log/pilot_transfer_selection}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKER_COUNT=8
CPU_THREADS_PER_WORKER="${CPU_THREADS_PER_WORKER:-24}"
MIN_HOST_RESERVE_GIB="${MIN_HOST_RESERVE_GIB:-256}"
PRELOAD_GIB_PER_WORKER="${PRELOAD_GIB_PER_WORKER:-125}"
MAX_EXISTING_GPU_MIB="${MAX_EXISTING_GPU_MIB:-1000}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
PROTOCOL_ID=pilot_transfer_selection_v1
PILOT_STEPS=(100 300 900)
ACTIVE_PIDS=()

if [[ "$DRY_RUN" == 1 ]]; then
  PYTHON="${PYTHON:-/opt/homebrew/bin/python3.11}"
else
  export PATH="/home/mhchu/miniconda3/bin:$PATH"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate prodigy
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"
fi

export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
export FINAL_CORE_CPU_THREADS="$CPU_THREADS_PER_WORKER"
export OMP_NUM_THREADS="$CPU_THREADS_PER_WORKER"
export MKL_NUM_THREADS="$CPU_THREADS_PER_WORKER"
export OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_WORKER"
export NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_WORKER"
export NUMEXPR_MAX_THREADS="$CPU_THREADS_PER_WORKER"

manifest_root="${EVAL_LOG_ROOT}/manifests/${RUN_ID}"
mkdir -p "$manifest_root"
"$PYTHON" "$SCRIPT_DIR/build_manifest.py" \
  --state-root "$TRAINING_STATE_ROOT" \
  --output "$manifest_root/stage_a_manifest.tsv" \
  --job-manifest-dir "$manifest_root/jobs"

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

wait_for_resources() {
  local required_gib=$((WORKER_COUNT * PRELOAD_GIB_PER_WORKER + MIN_HOST_RESERVE_GIB))
  while true; do
    local clear=1 gpu used available_kib available_gib
    for gpu in 0 1 2 3; do
      used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
      if (( used > MAX_EXISTING_GPU_MIB )); then
        echo "WAIT GPU $gpu busy: ${used} MiB" >&2
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
      echo "RESOURCE_GATE_OK host_gib=$available_gib utc=$(date -u +%FT%TZ)"
      return 0
    fi
    sleep 30
  done
}

check_checkpoints() {
  local step job_file checkpoint missing=0 count=0
  for step in "${PILOT_STEPS[@]}"; do
    job_file="$manifest_root/jobs/jobs_step${step}.json"
    while IFS= read -r checkpoint; do
      count=$((count + 1))
      if [[ ! -s "$checkpoint" ]]; then
        echo "MISSING $checkpoint" >&2
        missing=$((missing + 1))
      fi
    done < <("$PYTHON" -c \
      'import json,sys; print("\n".join(x["checkpoint"] for x in json.load(open(sys.argv[1]))))' \
      "$job_file")
  done
  [[ "$count" == 81 ]] || { echo "expected 81 checkpoint jobs, found $count" >&2; return 1; }
  [[ "$missing" == 0 ]] || { echo "$missing checkpoints missing" >&2; return 1; }
  echo "CHECKPOINT_PREFLIGHT_OK count=$count"
}

worker_command() {
  local step="$1" worker="$2" kind="$3" results_root="$4" ready_dir="$5"
  local -a cmd=(
    "$PYTHON" -u "$FINAL_CORE_DIR/evaluate_fixed_grid.py"
    --worker-index "$worker" --worker-count "$WORKER_COUNT"
    --partition-by targets --seeds 0,1,2
    --targets ukr_rus,covid,midterm,covid_political,election2020,ukr_rus_suspended,twibot20,cp_hk,facebook_page_reference
    --batch-size "$BATCH_SIZE" --episode-count 512
    --checkpoint-step "$step" --eval-split val --protocol "$PROTOCOL_ID"
    --config "$SCRIPT_DIR/evaluation.yaml"
    --training-state-root "$TRAINING_STATE_ROOT"
    --evaluation-state-root "$EVAL_STATE_ROOT/${kind}_step${step}"
    --evaluation-log-root "$EVAL_LOG_ROOT/internal/${kind}_step${step}"
    --results-root "$results_root"
    --evaluation-run-stamp "${RUN_ID}_${kind}_step${step}"
    --job-manifest "$manifest_root/jobs/jobs_step${step}.json"
    --ready-dir "$ready_dir" --expected-workers "$WORKER_COUNT"
    --min-host-reserve-gib "$MIN_HOST_RESERVE_GIB"
  )
  [[ "$kind" == smoke ]] && cmd+=(--max-checkpoints 1)
  printf '%q ' "${cmd[@]}"
  printf '\n'
}

launch_step() {
  local step="$1" kind="$2" results_root="$3" ready_dir="$4"
  local worker gpu status=0
  local -a pids=()
  mkdir -p "$results_root" "$ready_dir" "$EVAL_LOG_ROOT/queue"
  find "$ready_dir" -maxdepth 1 -type f -name 'worker_*.json' -delete
  for worker in 0 1 2 3 4 5 6 7; do
    gpu=$((worker / 2))
    echo "PLAN kind=$kind step=$step worker=$worker gpu=$gpu"
    if [[ "$DRY_RUN" == 1 ]]; then
      worker_command "$step" "$worker" "$kind" "$results_root" "$ready_dir"
      continue
    fi
    CUDA_VISIBLE_DEVICES="$gpu" bash -c \
      "$(worker_command "$step" "$worker" "$kind" "$results_root" "$ready_dir")" \
      > "$EVAL_LOG_ROOT/queue/${kind}_step${step}_worker${worker}.log" 2>&1 &
    pids+=("$!")
  done
  [[ "$DRY_RUN" == 1 ]] && return 0
  ACTIVE_PIDS=("${pids[@]}")
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  ACTIVE_PIDS=()
  return "$status"
}

if [[ "$DRY_RUN" != 1 ]]; then
  check_checkpoints
fi

if [[ "$SKIP_SMOKE" != 1 ]]; then
  smoke_root="$EVAL_LOG_ROOT/smoke/${RUN_ID}/step_300"
  smoke_ready="$EVAL_LOG_ROOT/ready/${RUN_ID}_smoke_step300"
  [[ "$DRY_RUN" == 1 ]] || wait_for_resources
  launch_step 300 smoke "$smoke_root" "$smoke_ready"
fi
[[ "$SMOKE_ONLY" == 1 ]] && exit 0

for step in "${PILOT_STEPS[@]}"; do
  results_root="$EVAL_LOG_ROOT/production/${RUN_ID}/results/step_${step}"
  ready_dir="$EVAL_LOG_ROOT/ready/${RUN_ID}_production_step${step}"
  [[ "$DRY_RUN" == 1 ]] || wait_for_resources
  launch_step "$step" production "$results_root" "$ready_dir"
done

if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY_RUN_COMPLETE expected_cells=729 checkpoint_jobs=81 workers=8 gpus=0,1,2,3"
  exit 0
fi

"$PYTHON" "$SCRIPT_DIR/aggregate_stage_a.py" \
  --results-root "$EVAL_LOG_ROOT/production/${RUN_ID}/results" \
  --manifest "$manifest_root/stage_a_manifest.tsv" \
  --output-root "$EVAL_LOG_ROOT/production/${RUN_ID}/summary" \
  --batch-size "$BATCH_SIZE"
echo "STAGE_A_EVALUATION_COMPLETE run_id=$RUN_ID"
