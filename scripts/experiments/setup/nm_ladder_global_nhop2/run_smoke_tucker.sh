#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PATH="/home/mhchu/miniconda3/bin:${PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=disabled
cd "${REPO_ROOT}"
"${CONDA_PREFIX}/bin/python" -u experiments/run_single_experiment.py \
  --config "${SCRIPT_DIR}/configs/smoke.yaml" --device 0 \
  --prefix finalcore_global_smoke_r2 --timestamp "${SMOKE_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}" \
  --state_dir "${REPO_ROOT}/state/nm_ladder_global_finalcore_smoke" \
  --log_dir "${REPO_ROOT}/log/nm_ladder_global_finalcore_smoke"
