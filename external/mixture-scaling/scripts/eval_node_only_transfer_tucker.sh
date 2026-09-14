#!/usr/bin/env bash
set -euo pipefail

OBJECTIVE="${1:?usage: $0 lp|fp}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_ROOT="${STATE_ROOT:-${ROOT}/state/node_only_transfer}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results/node_only_transfer}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/log/node_only_transfer/eval_${OBJECTIVE}}"
if [[ "$OBJECTIVE" != lp && "$OBJECTIVE" != fp ]]; then
  echo "objective must be lp or fp" >&2
  exit 2
fi
mkdir -p "$LOG_ROOT"
cd "$ROOT"

for gpu in 0 1 2 3; do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")"
  if (( used >= 2000 )); then
    echo "GPU $gpu is occupied: ${used} MiB" >&2
    exit 3
  fi
done

pids=()
for worker in 0 1 2 3; do
  gpu="$worker"
  PYTHONPATH=src python -m mixture_scaling.evaluate_node_only \
    --config configs/node_only_transfer.yaml --objective "$OBJECTIVE" \
    --worker-index "$worker" --workers 4 --device "$gpu" \
    --state-root "$STATE_ROOT" --output-root "$RESULTS_ROOT" --seed 0 \
    > "$LOG_ROOT/worker_${worker}_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status )); then exit 1; fi
PYTHONPATH=src python -m mixture_scaling.aggregate_node_only \
  --objective "$OBJECTIVE" --state-root "$STATE_ROOT" \
  --results-root "$RESULTS_ROOT" --output-root "$RESULTS_ROOT/aggregated"
date -u +%FT%TZ > "$LOG_ROOT/COMPLETE"
