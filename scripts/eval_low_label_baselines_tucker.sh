#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PY=/home/mhchu/miniconda3/envs/prodigy/bin/python3
CFG="${ROOT}/configs/twibot_strict_pilot.yaml"; SPLIT="${ROOT}/state/twibot_strict_pilot/splits"; OUT="${ROOT}/results/transfer_baselines_support10"; LOG="${ROOT}/log/transfer_baselines_support10"
mkdir -p "$OUT/raw" "$LOG"; JOBS=()
for target in twibot20 cora; do for feature in existing structural; do JOBS+=("$target|$feature|raw|0"); for seed in 0 1 2; do JOBS+=("$target|$feature|random_sage|$seed"); done; done; done
worker(){ local wi=$1 gpu=$2 i=0 job target feature baseline seed output; for job in "${JOBS[@]}"; do if ((i%8==wi)); then IFS='|' read -r target feature baseline seed <<<"$job"; output="$OUT/raw/${target}_${feature}_${baseline}_s${seed}.json"; if [[ ! -f "$output" ]]; then PYTHONPATH="$ROOT/src" "$PY" -u -m mixture_scaling.probe_low_label_baseline --config "$CFG" --split-root "$SPLIT" --target "$target" --feature-mode "$feature" --baseline "$baseline" --encoder-seed "$seed" --device "$gpu" --output "$output" >"$LOG/${target}_${feature}_${baseline}_s${seed}.log" 2>&1; fi; fi; ((i+=1)); done; }
pids=(); wi=0; for gpu in 2 3; do for slot in 0 1 2 3; do worker "$wi" "$gpu" & pids+=("$!"); ((wi+=1)); done; done
status=0; for pid in "${pids[@]}"; do wait "$pid"||status=1; done; ((status==0))||exit "$status"
echo "16 baseline cells complete"
