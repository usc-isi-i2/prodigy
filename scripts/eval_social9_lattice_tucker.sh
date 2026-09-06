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
