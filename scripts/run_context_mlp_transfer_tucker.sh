#!/usr/bin/env bash
set -euo pipefail
VIEW="${1:?usage: $0 neighborhood|node_neighborhood fp|lp [gate|full]}"; OBJECTIVE="${2:?}"; SELECTION="${3:-full}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; STATE_ROOT="${STATE_ROOT:-$ROOT/state/context_mlp_transfer}"; LOG_ROOT="${LOG_ROOT:-$ROOT/log/context_mlp_transfer/train_${VIEW}_${OBJECTIVE}_${SELECTION}}"; mkdir -p "$LOG_ROOT"; cd "$ROOT"
for gpu in 0 1 2 3; do used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")"; (( used < 2000 )) || { echo "GPU $gpu occupied" >&2; exit 3; }; done
for worker in 0 1 2 3; do PYTHONPATH=src python -m mixture_scaling.context_mlp_transfer --config configs/node_only_transfer.yaml --view "$VIEW" --objective "$OBJECTIVE" --selection "$SELECTION" --worker-index "$worker" --workers 4 --device "$worker" --output-root "$STATE_ROOT" --cache-root "$STATE_ROOT/_cache" --seed 0 >"$LOG_ROOT/worker_${worker}.log" 2>&1 & pids[$worker]=$!; done
status=0; for pid in "${pids[@]}"; do wait "$pid" || status=1; done; (( status==0 )) || exit 1; date -u +%FT%TZ >"$LOG_ROOT/COMPLETE"
