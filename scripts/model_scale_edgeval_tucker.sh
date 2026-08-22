#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/mhchu/miniconda3/envs/prodigy/bin/python3}"
SPLIT_ROOT="${SPLIT_ROOT:-${ROOT}/state/twibot_strict_pilot/splits}"
LADDER_CSV="${LADDER_CSV:-${ROOT}/results/edgeval_ladder_s0/ladder_auc.csv}"
STATE_ROOT="${STATE_ROOT:-${ROOT}/state/model_scale_edgeval_s0}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/model_scale_edgeval_s0/raw}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/log/model_scale_edgeval_s0}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-3}"
mkdir -p "${STATE_ROOT}" "${RESULT_ROOT}" "${LOG_ROOT}"

# Reuse the exact k=1,3,6 source subsets from the completed width-256 ladder.
# Width 256 is rerun only where that ladder used structural rather than existing
# features, giving a fully paired existing-feature comparison at all widths.
mapfile -t ROWS < <("${PYTHON_BIN}" - "${LADDER_CSV}" <<'PY'
import csv, sys
with open(sys.argv[1], newline="") as handle:
    for row in csv.DictReader(handle):
        if int(row["k"]) not in (1, 3, 6):
            continue
        for width in (64, 256, 512):
            if width == 256 and row["mode"] == "existing":
                continue
            run_id = f'mscale_{row["target"]}_k{row["k"]}_subset{row["subset"]}_w{width}_existing_s0'
            print("\t".join((run_id, row["target"], row["sources"], str(width))))
PY
)

TOTAL_WORKERS=$((2 * WORKERS_PER_GPU))
worker() {
  local worker_index="$1" gpu="$2" index=0 row run_id target sources width output
  for row in "${ROWS[@]}"; do
    if (( index % TOTAL_WORKERS == worker_index )); then
      IFS=$'\t' read -r run_id target sources width <<< "${row}"
      output="${RESULT_ROOT}/${run_id}.json"
      if [[ -f "${output}" ]]; then
        echo "[gpu ${gpu}] SKIP ${run_id}"
      else
        echo "[gpu ${gpu}] TRAIN ${run_id}"
        PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" -u -m mixture_scaling.train_strict \
          --config "${ROOT}/configs/twibot_strict_pilot.yaml" \
          --split-root "${SPLIT_ROOT}" --sources "${sources}" --run-id "${run_id}" \
          --device "${gpu}" --seed 0 --output-root "${STATE_ROOT}" \
          --hidden-dim "${width}" --output-dim "${width}" \
          >"${LOG_ROOT}/${run_id}.log" 2>&1
        echo "[gpu ${gpu}] PROBE ${run_id}"
        PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" -u -m mixture_scaling.probe_strict \
          --config "${ROOT}/configs/twibot_strict_pilot.yaml" \
          --split-root "${SPLIT_ROOT}" --checkpoint "${STATE_ROOT}/${run_id}/best.pt" \
          --target "${target}" --device "${gpu}" --seed 0 --selection-metric auc \
          --output "${output}" >"${LOG_ROOT}/${run_id}_probe.log" 2>&1
      fi
    fi
    ((index+=1))
  done
}

pids=()
worker_index=0
for gpu in 2 3; do
  for ((slot=0; slot<WORKERS_PER_GPU; slot++)); do
    worker "${worker_index}" "${gpu}" & pids+=("$!")
    ((worker_index+=1))
  done
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"
