#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_DIR="${RUN_DIR:-/dataMeR1/phil/gfm/prodigy-pinsage/log/pinsage_fast_$(date +%Y%m%d_%H%M%S)}"

export PATH="/home/mhchu/miniconda3/bin:${PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
cd "${REPO_ROOT}"

args=(
  --configs "${SCRIPT_DIR}"/configs/train_*.yaml
  --gpus 0 1 2 3 --models-per-gpu 2 --worker-budget 32
  --run-dir "${RUN_DIR}"
)
[[ "${DRY_RUN:-0}" == 1 ]] && args+=(--dry-run)
[[ -n "${SMOKE_STEPS:-}" ]] && args+=(--smoke-steps "${SMOKE_STEPS}")
python experiments/run_shared_graph.py "${args[@]}"
