#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CONFIG_DIR="${CONFIG_DIR:-${SCRIPT_DIR}/configs}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/log/nm_pairwise_finalcore/shared_seed0_20260904}"
DRY_RUN="${DRY_RUN:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
MODELS_PER_GPU="${MODELS_PER_GPU:-8}"
WORKER_BUDGET="${WORKER_BUDGET:-128}"
THREADS_PER_MODEL="${THREADS_PER_MODEL:-4}"
MIN_HOST_AVAILABLE_GIB="${MIN_HOST_AVAILABLE_GIB:-512}"
MIN_SHM_AVAILABLE_GIB="${MIN_SHM_AVAILABLE_GIB:-200}"
MAX_EXISTING_GPU_MIB="${MAX_EXISTING_GPU_MIB:-1000}"

[[ "$DRY_RUN" =~ ^(0|1)$ ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
[[ "$PREFLIGHT_ONLY" =~ ^(0|1)$ ]] || { echo "PREFLIGHT_ONLY must be 0 or 1" >&2; exit 2; }
[[ "$MODELS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || { echo "MODELS_PER_GPU must be positive" >&2; exit 2; }

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONDONTWRITEBYTECODE=1
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"
unset CUDA_VISIBLE_DEVICES

cd "$REPO_ROOT"
mapfile -t CONFIGS < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name '*.yaml' | sort)
[[ "${#CONFIGS[@]}" == 36 ]] || {
  echo "expected 36 generated pair configs in $CONFIG_DIR; run make_configs.py" >&2
  exit 2
}
[[ "$($PYTHON "$SCRIPT_DIR/pair_plan.py" | awk 'END {print NR-1}')" == 36 ]] || {
  echo "pair plan must contain exactly 36 models" >&2
  exit 2
}

if [[ "$DRY_RUN" != 1 ]]; then
  for gpu in 0 1 2 3; do
    used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    (( used <= MAX_EXISTING_GPU_MIB )) || {
      echo "GPU $gpu is busy (${used} MiB > ${MAX_EXISTING_GPU_MIB} MiB); refusing launch" >&2
      exit 1
    }
  done
  available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  available_gib=$((available_kib / 1024 / 1024))
  (( available_gib >= MIN_HOST_AVAILABLE_GIB )) || {
    echo "need ${MIN_HOST_AVAILABLE_GIB} GiB host RAM; only ${available_gib} GiB available" >&2
    exit 1
  }
  shm_available_kib="$(df --output=avail -k /dev/shm | tail -1 | tr -d ' ')"
  shm_available_gib=$((shm_available_kib / 1024 / 1024))
  (( shm_available_gib >= MIN_SHM_AVAILABLE_GIB )) || {
    echo "need ${MIN_SHM_AVAILABLE_GIB} GiB /dev/shm; only ${shm_available_gib} GiB available" >&2
    exit 1
  }
fi

cmd=("$PYTHON" -u experiments/run_shared_graph.py
     --configs "${CONFIGS[@]}"
     --gpus 0 1 2 3
     --models-per-gpu "$MODELS_PER_GPU"
     --worker-budget "$WORKER_BUDGET"
     --threads-per-model "$THREADS_PER_MODEL"
     --run-dir "$RUN_DIR")
[[ "$DRY_RUN" == 1 ]] && cmd+=(--dry-run)
[[ "$PREFLIGHT_ONLY" == 1 ]] && cmd+=(--preflight-only)
exec "${cmd[@]}"
