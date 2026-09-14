#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/mhchu/miniconda3/envs/prodigy/bin/python3}"
SPLIT_ROOT="${SPLIT_ROOT:-${ROOT}/state/twibot_strict_pilot/splits}"
LADDER_CSV="${LADDER_CSV:-${ROOT}/results/edgeval_ladder_s0/ladder_auc.csv}"
TARGET="${TARGET:-covid_political}"
RUN_PREFIX="${RUN_PREFIX:-covid}"
STATE_ROOT="${STATE_ROOT:-${ROOT}/state/${RUN_PREFIX}_model_scale_seeds}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/${RUN_PREFIX}_model_scale_seeds/raw}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/log/${RUN_PREFIX}_model_scale_seeds}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-3}"
mkdir -p "${STATE_ROOT}" "${RESULT_ROOT}" "${LOG_ROOT}"

mapfile -t ROWS < <("${PYTHON_BIN}" - "${LADDER_CSV}" "${TARGET}" "${RUN_PREFIX}" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1]))
        if r["target"] == sys.argv[2] and int(r["k"]) in (1, 6)]
for row in rows:
    for width in (64, 512):
        for seed in (1, 2):
            run_id = f'{sys.argv[3]}_k{row["k"]}_subset{row["subset"]}_w{width}_existing_s{seed}'
            print("\t".join((run_id, row["sources"], str(width), str(seed))))
PY
)

TOTAL_WORKERS=$((2 * WORKERS_PER_GPU))
worker() {
  local worker_index="$1" gpu="$2" index=0 row run_id sources width seed output
  for row in "${ROWS[@]}"; do
    if (( index % TOTAL_WORKERS == worker_index )); then
      IFS=$'\t' read -r run_id sources width seed <<< "${row}"
      sources="${sources//|/,}"
      output="${RESULT_ROOT}/${run_id}.json"
      if [[ -f "${output}" ]]; then
        echo "[gpu ${gpu}] SKIP ${run_id}"
      else
        if [[ ! -f "${STATE_ROOT}/${run_id}/summary.json" ]]; then
          echo "[gpu ${gpu}] TRAIN ${run_id}"
          PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" -u -m mixture_scaling.train_strict \
            --config "${ROOT}/configs/twibot_strict_pilot.yaml" --split-root "${SPLIT_ROOT}" \
            --sources "${sources}" --run-id "${run_id}" --device "${gpu}" --seed "${seed}" \
            --output-root "${STATE_ROOT}" --hidden-dim "${width}" --output-dim "${width}" \
            >"${LOG_ROOT}/${run_id}.log" 2>&1
        fi
        echo "[gpu ${gpu}] PROBE ${run_id}"
        PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" -u -m mixture_scaling.probe_strict \
          --config "${ROOT}/configs/twibot_strict_pilot.yaml" --split-root "${SPLIT_ROOT}" \
          --checkpoint "${STATE_ROOT}/${run_id}/best.pt" --target "${TARGET}" \
          --device "${gpu}" --seed "${seed}" --selection-metric auc --output "${output}" \
          >"${LOG_ROOT}/${run_id}_probe.log" 2>&1
      fi
    fi
    ((index+=1))
  done
}

pids=(); worker_index=0
for gpu in 2 3; do
  for ((slot=0; slot<WORKERS_PER_GPU; slot++)); do
    worker "${worker_index}" "${gpu}" & pids+=("$!"); ((worker_index+=1))
  done
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"
