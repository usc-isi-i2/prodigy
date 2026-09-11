#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH=src
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-node-mlp-ladder}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STATE_ROOT="${STATE_ROOT:-/dataMeR1/phil/gfm/mixture-scaling/state/node_mlp_ladder_s0_2500}"
RESULTS_ROOT="${RESULTS_ROOT:-/dataMeR1/phil/gfm/mixture-scaling/results/node_mlp_ladder_s0_2500/raw}"
LOG_ROOT="${LOG_ROOT:-/dataMeR1/phil/gfm/mixture-scaling/log/node_mlp_ladder_s0_2500}"
CACHE_ROOT="${CACHE_ROOT:-/dataMeR1/phil/gfm/mixture-scaling-node-only/state/node_only_transfer/_cache}"
for gpu in 2 3; do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")"
  if (( used >= 2000 )); then echo "GPU $gpu occupied: ${used} MiB" >&2; exit 3; fi
done
if [[ -e "$LOG_ROOT" ]]; then echo "Log directory exists; choose a fresh LOG_ROOT" >&2; exit 4; fi
mkdir -p "$STATE_ROOT" "$RESULTS_ROOT" "$LOG_ROOT"
if [[ -e "$STATE_ROOT/_cache" || -L "$STATE_ROOT/_cache" ]]; then
  [[ "$(readlink -f "$STATE_ROOT/_cache")" == "$(readlink -f "$CACHE_ROOT")" ]] || exit 5
else
  ln -s "$CACHE_ROOT" "$STATE_ROOT/_cache"
fi
git rev-parse HEAD > "$LOG_ROOT/revision.txt"
date -u +%FT%TZ > "$LOG_ROOT/STARTED"
args=(--state-root "$STATE_ROOT" --output-root "$RESULTS_ROOT" --cache-root "$CACHE_ROOT")
python -m mixture_scaling.node_mlp_ladder plan "${args[@]}" > "$LOG_ROOT/plan.json"
for phase in train eval; do
  pids=()
  for worker in 0 1; do
    python -u -m mixture_scaling.node_mlp_ladder "$phase" "${args[@]}" \
      --worker-index "$worker" --workers 2 --device "$((worker + 2))" \
      > "$LOG_ROOT/${phase}_${worker}.log" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  if (( status )); then echo "$phase failed" > "$LOG_ROOT/FAILED"; exit 1; fi
done
python -m mixture_scaling.node_mlp_ladder aggregate "${args[@]}" > "$LOG_ROOT/aggregate.log" 2>&1
date -u +%FT%TZ > "$LOG_ROOT/COMPLETE"
