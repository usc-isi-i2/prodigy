#!/usr/bin/env bash
set -euo pipefail

OBJECTIVE="${1:?usage: $0 lp|graphmae cls|lp}"
TASK="${2:?usage: $0 lp|graphmae cls|lp}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_ROOT="${STATE_ROOT:-${ROOT}/state/social9_source_lattice}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/social9_source_lattice/${OBJECTIVE}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/log/social9_source_lattice/eval_${OBJECTIVE}_${TASK}}"
mkdir -p "$LOG_ROOT"
cd "$ROOT"

if [[ "$OBJECTIVE" != lp && "$OBJECTIVE" != graphmae ]]; then
  echo "objective must be lp or graphmae" >&2
  exit 2
fi
if [[ "$TASK" != cls && "$TASK" != lp ]]; then
  echo "task must be cls or lp" >&2
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
  echo "task=$TASK"
  echo "seed=0"
  echo "physical_gpus=0,1,2,3"
  echo "wandb_mode=offline"
  echo "state_root=$STATE_ROOT"
  echo "output_root=$OUTPUT_ROOT"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$LOG_ROOT/provenance.txt"

pids=()
for gpu in 0 1 2 3; do
  WANDB_MODE=offline PYTHONPATH=src python -m mixture_scaling.evaluate_lattice \
    --config configs/social9_lattice.yaml --objective "$OBJECTIVE" --task "$TASK" \
    --worker-index "$gpu" --workers 4 --device "$gpu" \
    --state-root "$STATE_ROOT" --output-root "$OUTPUT_ROOT" --seed 0 \
    > "$LOG_ROOT/worker_${gpu}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status )); then exit 1; fi
date -u +%FT%TZ > "$LOG_ROOT/COMPLETE"
