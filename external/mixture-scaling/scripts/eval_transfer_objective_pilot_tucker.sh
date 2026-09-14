#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; PY="/home/mhchu/miniconda3/envs/prodigy/bin/python3"
CFG="${ROOT}/configs/twibot_strict_pilot.yaml"; SPLIT="${ROOT}/state/twibot_strict_pilot/splits"
STATE="${ROOT}/state/transfer_objective_pilot"; RESULT="${ROOT}/results/transfer_objective_pilot"; LOG="${ROOT}/log/transfer_objective_pilot_eval"
mkdir -p "${RESULT}/raw" "${LOG}"; JOBS=()
for objective in link graphmae; do for feature in existing structural; do for seed in 0 1 2; do
  JOBS+=("${objective}|${feature}|${seed}|target|twibot20|target_twibot" "${objective}|${feature}|${seed}|target|cora|target_cora")
  JOBS+=("${objective}|${feature}|${seed}|cross|twibot20|cross_election" "${objective}|${feature}|${seed}|cross|cora|cross_election")
  JOBS+=("${objective}|${feature}|${seed}|mixture|twibot20|loo_twibot" "${objective}|${feature}|${seed}|mixture|cora|loo_cora")
done; done; done
worker() { local wi="$1" gpu="$2" i=0 job objective feature seed condition target key run output
  for job in "${JOBS[@]}"; do if (( i % 8 == wi )); then IFS='|' read -r objective feature seed condition target key <<<"${job}"
    run="${objective}_${feature}_${key}_s${seed}"; output="${RESULT}/raw/${objective}_${feature}_${condition}_${target}_s${seed}.json"
    if [[ ! -f "$output" ]]; then PYTHONPATH="${ROOT}/src" "$PY" -u -m mixture_scaling.probe_label_budget --config "$CFG" --split-root "$SPLIT" --checkpoint "${STATE}/${run}/best.pt" --target "$target" --condition "$condition" --device "$gpu" --seed "$seed" --output "$output" >"${LOG}/${objective}_${feature}_${condition}_${target}_s${seed}.log" 2>&1; fi
  fi; ((i+=1)); done; }
pids=(); wi=0; for gpu in 2 3; do for slot in 0 1 2 3; do worker "$wi" "$gpu" & pids+=("$!"); ((wi+=1)); done; done
status=0; for pid in "${pids[@]}"; do wait "$pid" || status=1; done; (( status == 0 )) || exit "$status"
PYTHONPATH="${ROOT}/src" "$PY" -m mixture_scaling.aggregate_transfer_pilot --raw-root "${RESULT}/raw" --output "${RESULT}/low_label_auc.csv"
