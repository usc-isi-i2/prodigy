#!/usr/bin/env bash
set -euo pipefail
VIEW="${1:?usage: $0 neighborhood|node_neighborhood fp|lp}"; OBJECTIVE="${2:?}"; ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; STATE_ROOT="${STATE_ROOT:-$ROOT/state/context_mlp_transfer}"; RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/results/context_mlp_transfer}"; LOG_ROOT="${LOG_ROOT:-$ROOT/log/context_mlp_transfer/eval_${VIEW}_${OBJECTIVE}}"; mkdir -p "$LOG_ROOT"; cd "$ROOT"
FANOUT_ARGS=(); [[ -z "${FANOUT:-}" ]] || FANOUT_ARGS=(--fanout "$FANOUT")
for worker in 0 1 2 3; do PYTHONPATH=src python -m mixture_scaling.evaluate_context_mlp --config configs/context_mlp_transfer.yaml --view "$VIEW" --objective "$OBJECTIVE" --worker-index "$worker" --workers 4 --device "$worker" --state-root "$STATE_ROOT" --output-root "$RESULTS_ROOT" --seed 0 "${FANOUT_ARGS[@]}" >"$LOG_ROOT/worker_${worker}.log" 2>&1 & pids[$worker]=$!; done
status=0; for pid in "${pids[@]}"; do wait "$pid" || status=1; done; (( status==0 )) || exit 1
PYTHONPATH=src python -m mixture_scaling.aggregate_context_mlp --view "$VIEW" --objective "$OBJECTIVE" --state-root "$STATE_ROOT" --results-root "$RESULTS_ROOT" --output-root "$RESULTS_ROOT/aggregated"; date -u +%FT%TZ >"$LOG_ROOT/COMPLETE"
