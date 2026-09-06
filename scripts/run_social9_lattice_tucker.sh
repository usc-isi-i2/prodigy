#!/usr/bin/env bash
set -euo pipefail

OBJECTIVE="${1:?usage: $0 lp|graphmae [gate|full]}"
SELECTION="${2:-full}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/state/social9_source_lattice}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/log/social9_source_lattice/${OBJECTIVE}_${SELECTION}}"
mkdir -p "$LOG_ROOT"
cd "$ROOT"

if [[ "$OBJECTIVE" != lp && "$OBJECTIVE" != graphmae ]]; then
  echo "objective must be lp or graphmae" >&2
  exit 2
fi

for gpu in 0 1 2 3; do
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
  echo "physical_gpus=0,1,2,3"
  echo "wandb_mode=offline"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$LOG_ROOT/provenance.txt"

pids=()
for gpu in 0 1 2 3; do
  WANDB_MODE=offline PYTHONPATH=src python -m mixture_scaling.lattice \
    --config configs/social9_lattice.yaml \
    --objective "$OBJECTIVE" --selection "$SELECTION" \
    --worker-index "$gpu" --workers 4 --device "$gpu" \
    --output-root "$OUTPUT_ROOT" --seed 0 \
    > "$LOG_ROOT/worker_${gpu}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if (( status != 0 )); then
  echo "one or more workers failed; inspect $LOG_ROOT" >&2
  exit 1
fi
date -u +%FT%TZ > "$LOG_ROOT/COMPLETE"
