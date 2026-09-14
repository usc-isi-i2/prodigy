#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/mhchu/miniconda3/envs/prodigy/bin/python3}"
MANIFEST="${MANIFEST:-${ROOT}/manifests/primary.tsv}"
STATE_ROOT="${STATE_ROOT:-/dataMeR1/phil/gfm/mixture-scaling/state/primary_s0}"
RESULT_ROOT="${RESULT_ROOT:-/dataMeR1/phil/gfm/mixture-scaling/results/primary_s0/raw}"
LOG_ROOT="${LOG_ROOT:-/dataMeR1/phil/gfm/mixture-scaling/log/primary_s0_eval}"
GPUS_TEXT="${GPUS:-2 3}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
STEPS_TEXT="${STEPS:-100 300 900 2500}"
mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}"
read -r -a GPU_IDS <<< "${GPUS_TEXT}"
read -r -a STEPS <<< "${STEPS_TEXT}"
for gpu in "${GPU_IDS[@]}"; do
  [[ "${gpu}" == 2 || "${gpu}" == 3 ]] || { echo "forbidden GPU: ${gpu}" >&2; exit 2; }
done

jobs=()
while IFS=$'\t' read -r run_id kind target sources seed; do
  [[ "${run_id}" == run_id ]] && continue
  for step in "${STEPS[@]}"; do jobs+=("${run_id}"$'\t'"${target}"$'\t'"${step}"); done
done < "${MANIFEST}"
TOTAL_WORKERS=$(( ${#GPU_IDS[@]} * WORKERS_PER_GPU ))

worker() {
  local worker_index="$1" gpu="$2" index=0 job run_id target step checkpoint output
  for job in "${jobs[@]}"; do
    if (( index % TOTAL_WORKERS == worker_index )); then
      IFS=$'\t' read -r run_id target step <<< "${job}"
      checkpoint="${STATE_ROOT}/${run_id}/checkpoints/step_${step}.pt"
      output="${RESULT_ROOT}/${run_id}__${target}__step${step}.json"
      [[ -f "${checkpoint}" ]] || { echo "missing ${checkpoint}" >&2; return 1; }
      if [[ -f "${output}" ]]; then
        echo "[gpu ${gpu}] SKIP ${run_id} ${target} step=${step}"
      else
        echo "[gpu ${gpu}] EVAL ${run_id} ${target} step=${step}"
        PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" -u -m mixture_scaling.evaluate \
          --config "${ROOT}/configs/graphs.yaml" --checkpoint "${checkpoint}" \
          --target "${target}" --device "${gpu}" --output "${output}" \
          >"${LOG_ROOT}/${run_id}__${target}__step${step}.log" 2>&1
      fi
    fi
    ((index+=1))
  done
}

pids=()
worker_index=0
for gpu in "${GPU_IDS[@]}"; do
  for (( slot=0; slot<WORKERS_PER_GPU; slot++ )); do
    worker "${worker_index}" "${gpu}" & pids+=("$!")
    ((worker_index+=1))
  done
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"
