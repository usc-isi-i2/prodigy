#!/usr/bin/env bash
# Replicate the matched flagship intervention ladders at seeds 1 and 2.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CAMPAIGN_ROOT="${REPO_ROOT}/scripts/experiments/setup/nm_interventions_overnight/configs"
SEEDS_TEXT="${SEEDS:-1 2}"
GPUS_TEXT="${GPUS:-0 2 3}"
MODELS_PER_GPU="${MODELS_PER_GPU:-6}"
WORKER_BUDGET="${WORKER_BUDGET:-72}"
RUN_STAMP="${RUN_STAMP:-20260908}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/log/paper_flagship_ladders/${RUN_STAMP}}"
ARMS_TEXT="${ARMS:-baseline objective exposure schedule composition capacity}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
unset CUDA_VISIBLE_DEVICES || true
cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT"

configs=()
arm_count=0
for arm in $ARMS_TEXT; do
  arm_count=$((arm_count + 1))
  for rung in {1..8}; do
    config="$CAMPAIGN_ROOT/${arm}_r${rung}_s0.yaml"
    [[ -f "$config" ]] || { echo "missing $config" >&2; exit 2; }
    configs+=("$config")
  done
done
expected_configs=$((arm_count * 8))
[[ ${#configs[@]} -eq "$expected_configs" ]] || {
  echo "expected $expected_configs flagship configs" >&2
  exit 2
}

run_dir="$RUN_ROOT/seeds_$(tr ' ' '-' <<< "$SEEDS_TEXT")"
if [[ -f "$run_dir/status.json" ]] && grep -q '"status": "complete"' "$run_dir/status.json"; then
  echo "SKIP complete seeds=$SEEDS_TEXT"
  exit 0
fi
[[ ! -e "$run_dir" ]] || { echo "REFUSE existing incomplete $run_dir" >&2; exit 1; }
mode_args=()
[[ "${DRY_RUN:-0}" == 1 ]] && mode_args+=(--dry-run)
"${CONDA_PREFIX}/bin/python" experiments/run_shared_graph.py \
  --configs "${configs[@]}" --seeds $SEEDS_TEXT --gpus $GPUS_TEXT \
  --models-per-gpu "$MODELS_PER_GPU" --worker-budget "$WORKER_BUDGET" \
  --threads-per-model 4 --run-dir "$run_dir" "${mode_args[@]}"

[[ "${DRY_RUN:-0}" == 1 ]] || date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/training_complete_utc.txt"
