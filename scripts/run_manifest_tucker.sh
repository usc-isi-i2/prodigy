#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${MANIFEST:?set MANIFEST to a training TSV}"
STATE_ROOT="${STATE_ROOT:?set STATE_ROOT to a new output directory}"
LOG_ROOT="${LOG_ROOT:?set LOG_ROOT to a new log directory}"
GPUS_TEXT="${GPUS:-2 3}"
mkdir -p "${STATE_ROOT}" "${LOG_ROOT}"
read -r -a GPU_IDS <<< "${GPUS_TEXT}"
for gpu in "${GPU_IDS[@]}"; do
  [[ "${gpu}" == 2 || "${gpu}" == 3 ]] || { echo "forbidden GPU: ${gpu}" >&2; exit 2; }
done
mapfile -t ROWS < <(tail -n +2 "${MANIFEST}")

worker() {
  local worker_index="$1" gpu="$2" index=0 row run_id kind target sources seed summary
  for row in "${ROWS[@]}"; do
    if (( index % ${#GPU_IDS[@]} == worker_index )); then
      IFS=$'\t' read -r run_id kind target sources seed <<< "${row}"
      summary="${STATE_ROOT}/${run_id}/summary.json"
      if [[ -f "${summary}" ]]; then
        echo "[gpu ${gpu}] SKIP ${run_id}"
      else
        echo "[gpu ${gpu}] START ${run_id} sources=${sources}"
        PYTHONPATH="${ROOT}/src" python3 -u -m mixture_scaling.train \
          --config "${ROOT}/configs/graphs.yaml" --sources "${sources}" \
          --run-id "${run_id}" --seed "${seed}" --device "${gpu}" \
          --output-root "${STATE_ROOT}" >"${LOG_ROOT}/${run_id}.log" 2>&1
      fi
    fi
    ((index+=1))
  done
}

pids=()
for index in "${!GPU_IDS[@]}"; do worker "${index}" "${GPU_IDS[$index]}" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"
