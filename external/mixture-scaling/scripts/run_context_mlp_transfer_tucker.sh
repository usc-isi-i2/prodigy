#!/usr/bin/env bash
set -euo pipefail
VIEW="${1:?usage: $0 neighborhood|node_neighborhood fp|lp [gate|full]}"; OBJECTIVE="${2:?}"; SELECTION="${3:-full}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; STATE_ROOT="${STATE_ROOT:-$ROOT/state/context_mlp_transfer}"; LOG_ROOT="${LOG_ROOT:-$ROOT/log/context_mlp_transfer/train_${VIEW}_${OBJECTIVE}_${SELECTION}}"; mkdir -p "$LOG_ROOT"; cd "$ROOT"
FANOUT_ARGS=(); [[ -z "${FANOUT:-}" ]] || FANOUT_ARGS=(--fanout "$FANOUT")
read -r -a devices <<< "${GPU_DEVICES:-0 1 2 3}"; workers="${#devices[@]}"
for gpu in "${devices[@]}"; do [[ "$gpu" =~ ^[0-3]$ ]] || { echo "invalid GPU $gpu" >&2; exit 2; }; used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")"; (( used < 2000 )) || { echo "GPU $gpu occupied" >&2; exit 3; }; done
for worker in "${!devices[@]}"; do gpu="${devices[$worker]}"; PYTHONPATH=src python -m mixture_scaling.context_mlp_transfer --config configs/context_mlp_transfer.yaml --view "$VIEW" --objective "$OBJECTIVE" --selection "$SELECTION" --worker-index "$worker" --workers "$workers" --device "$gpu" --output-root "$STATE_ROOT" --cache-root "$STATE_ROOT/_cache" --seed 0 "${FANOUT_ARGS[@]}" >"$LOG_ROOT/worker_${worker}.log" 2>&1 & pids[$worker]=$!; done
status=0; for pid in "${pids[@]}"; do wait "$pid" || status=1; done; (( status==0 )) || exit 1; date -u +%FT%TZ >"$LOG_ROOT/COMPLETE"
