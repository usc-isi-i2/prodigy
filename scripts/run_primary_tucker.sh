#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${MANIFEST:-${ROOT}/manifests/primary.tsv}"
STATE_ROOT="${STATE_ROOT:-/dataMeR1/phil/gfm/mixture-scaling/state/primary_s0}"
LOG_ROOT="${LOG_ROOT:-/dataMeR1/phil/gfm/mixture-scaling/log/primary_s0}"
GPUS_TEXT="${GPUS:-2 3}"
mkdir -p "${STATE_ROOT}" "${LOG_ROOT}"
read -r -a GPU_IDS <<< "${GPUS_TEXT}"
[[ " ${GPUS_TEXT} " == *" 2 "* || " ${GPUS_TEXT} " == *" 3 "* ]] || {
  echo "only GPUs 2 and 3 are allowed" >&2; exit 2;
}
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
        echo "[gpu ${gpu}] DONE ${run_id}"
      fi
    fi
    ((index+=1))
  done
}

pids=()
for index in "${!GPU_IDS[@]}"; do
  worker "${index}" "${GPU_IDS[$index]}" & pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"

