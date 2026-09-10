#!/usr/bin/env bash
set -euo pipefail

OBJECTIVE="${1:?usage: $0 lp|fp [gate|full]}"
SELECTION="${2:-full}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_ROOT="${STATE_ROOT:-${ROOT}/state/node_only_transfer}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/log/node_only_transfer/train_${OBJECTIVE}_${SELECTION}}"
if [[ "$OBJECTIVE" != lp && "$OBJECTIVE" != fp ]]; then
  echo "objective must be lp or fp" >&2
  exit 2
fi
mkdir -p "$LOG_ROOT"
cd "$ROOT"

for gpu in 2 3; do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")"
  if (( used >= 2000 )); then
    echo "GPU $gpu is occupied: ${used} MiB" >&2
    exit 3
  fi
done

{
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "objective=$OBJECTIVE"
  echo "selection=$SELECTION"
  echo "seed=0"
  echo "physical_gpus=2,3"
  echo "state_root=$STATE_ROOT"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$LOG_ROOT/provenance.txt"

pids=()
for worker in 0 1; do
  gpu=$((worker + 2))
  WANDB_MODE=offline PYTHONPATH=src python -m mixture_scaling.node_only_transfer \
    --config configs/node_only_transfer.yaml --objective "$OBJECTIVE" \
    --selection "$SELECTION" --worker-index "$worker" --workers 2 --device "$gpu" \
    --output-root "$STATE_ROOT" --cache-root "$STATE_ROOT/_cache" --seed 0 \
    > "$LOG_ROOT/worker_${worker}_gpu_${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status )); then exit 1; fi
date -u +%FT%TZ > "$LOG_ROOT/COMPLETE"
