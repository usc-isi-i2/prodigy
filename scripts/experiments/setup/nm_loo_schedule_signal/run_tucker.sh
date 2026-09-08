#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
STAMP="${RUN_STAMP:-20260908_signal1}"
RUN_DIR="$ROOT/log/nm_loo_schedule_signal/shared_${STAMP}"
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
cd "$ROOT"
python "$HERE/make_configs.py"
python experiments/run_shared_graph.py \
  --configs "$HERE"/configs/*.yaml \
  --gpus ${GPUS:-2 3} --models-per-gpu ${MODELS_PER_GPU:-2} \
  --worker-budget ${WORKER_BUDGET:-16} --run-dir "$RUN_DIR" ${DRY_RUN:+--dry-run}
