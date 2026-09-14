#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/mhchu/miniconda3/envs/prodigy/bin/python3}"
CONFIG="${ROOT}/configs/twibot_strict_pilot.yaml"
SPLIT_ROOT="${ROOT}/state/twibot_strict_pilot/splits"
STATE_ROOT="${ROOT}/state/transfer_objective_pilot"
LOG_ROOT="${ROOT}/log/transfer_objective_pilot"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
mkdir -p "${STATE_ROOT}" "${LOG_ROOT}"

SOURCE_KEYS=(target_twibot target_cora cross_election loo_twibot loo_cora)
declare -A SOURCES
SOURCES[target_twibot]="twibot20"
SOURCES[target_cora]="cora"
SOURCES[cross_election]="election2020"
SOURCES[loo_twibot]="covid_political,ukr_rus_suspended,election2020,facebook_page_reference,cora,pubmed"
SOURCES[loo_cora]="covid_political,ukr_rus_suspended,election2020,twibot20,facebook_page_reference,pubmed"

JOBS=()
for objective in link graphmae; do
  for feature_mode in existing structural; do
    for source_key in "${SOURCE_KEYS[@]}"; do
      for seed in 0 1 2; do
        JOBS+=("${objective}|${feature_mode}|${source_key}|${seed}")
      done
    done
  done
done

TOTAL_WORKERS=$((2 * WORKERS_PER_GPU))
worker() {
  local worker_index="$1" gpu="$2" index=0 job objective feature_mode source_key seed run_id
  for job in "${JOBS[@]}"; do
    if (( index % TOTAL_WORKERS == worker_index )); then
      IFS='|' read -r objective feature_mode source_key seed <<<"${job}"
      run_id="${objective}_${feature_mode}_${source_key}_s${seed}"
      if [[ ! -f "${STATE_ROOT}/${run_id}/summary.json" ]]; then
        echo "[gpu ${gpu}] ${run_id}"
        PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" -u -m mixture_scaling.train_transfer_pilot \
          --config "${CONFIG}" --split-root "${SPLIT_ROOT}" \
          --sources "${SOURCES[$source_key]}" --objective "${objective}" \
          --feature-mode "${feature_mode}" --run-id "${run_id}" \
          --device "${gpu}" --seed "${seed}" --output-root "${STATE_ROOT}" \
          >"${LOG_ROOT}/${run_id}.log" 2>&1
      fi
    fi
    ((index+=1))
  done
}

pids=(); worker_index=0
for gpu in 2 3; do
  for ((slot=0; slot<WORKERS_PER_GPU; slot++)); do
    worker "${worker_index}" "${gpu}" & pids+=("$!")
    ((worker_index+=1))
  done
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 )) || exit "${status}"

echo "all 60 distinct encoders complete"
