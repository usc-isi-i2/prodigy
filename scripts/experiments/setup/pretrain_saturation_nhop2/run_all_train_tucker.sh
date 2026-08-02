#!/usr/bin/env bash
# Train the three fresh trajectories. One process per GPU, one graph load per process.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/run_logs"
mkdir -p "${LOG_DIR}"

CONFIGS=(train_all8.yaml train_covid.yaml train_ukr.yaml)
read -r -a GPU_ARR <<< "${GPUS:-0}"
[[ ${#GPU_ARR[@]} -gt 0 ]] || { echo "need at least one GPU" >&2; exit 2; }
stamp="$(date +%Y%m%d_%H%M%S)"

declare -a BUCKET
for i in "${!CONFIGS[@]}"; do
  slot=$((i % ${#GPU_ARR[@]}))
  BUCKET[$slot]+="${CONFIGS[$i]}"$'\n'
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  for slot in "${!GPU_ARR[@]}"; do
    while IFS= read -r config; do
      [[ -n "${config}" ]] || continue
      DRY_RUN=1 "${SCRIPT_DIR}/train_nm_tucker.sh" "${config}" \
        --device "${GPU_ARR[$slot]}" "$@"
    done <<< "${BUCKET[$slot]:-}"
  done
  exit 0
fi

declare -a PIDS USED
for slot in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$slot]}"
  bucket="${BUCKET[$slot]:-}"
  [[ -n "${bucket//[$'\n']/}" ]] || continue
  (
    while IFS= read -r config; do
      [[ -n "${config}" ]] || continue
      name="${config%.yaml}"
      log="${LOG_DIR}/${name}_gpu${gpu}_${stamp}.log"
      echo "[gpu ${gpu}] launching ${name} -> ${log}" >&2
      "${SCRIPT_DIR}/train_nm_tucker.sh" "${config}" --device "${gpu}" "$@" \
        >"${log}" 2>&1
      echo "[gpu ${gpu}] completed ${name}" >&2
    done <<< "${bucket}"
  ) &
  PIDS+=("$!"); USED+=("${gpu}")
done

rc=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}" || { echo "worker on GPU ${USED[$i]} failed" >&2; rc=1; }
done
exit "${rc}"
