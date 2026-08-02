#!/usr/bin/env bash
# Run all eight canonical sequential rungs, or the merged-graph schedule smoke.
#   PHASE=smoke GPUS="0" bash run_all_train_tucker.sh
#   PHASE=all GPUS="0" bash run_all_train_tucker.sh
# Defaults to one GPU because every process loads the ~104 GB all8 graph.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/run_logs"
mkdir -p "${LOG_DIR}"

case "${PHASE:-all}" in
  smoke) ALL_CONFIGS=(smoke.yaml) ;;
  all) ALL_CONFIGS=(train_r1.yaml train_r2.yaml train_r3.yaml train_r4.yaml
                      train_r5.yaml train_r6.yaml train_r7.yaml train_r8.yaml) ;;
  *) echo "PHASE must be smoke or all" >&2; exit 2 ;;
esac

is_skipped() { [[ " ${SKIP:-} " == *" $1 "* ]]; }
CONFIGS=()
for config in "${ALL_CONFIGS[@]}"; do
  is_skipped "${config}" && { echo "skipping ${config}" >&2; continue; }
  [[ -f "${SCRIPT_DIR}/configs/${config}" ]] || {
    echo "missing ${config}; run make_configs.py" >&2
    exit 2
  }
  CONFIGS+=("${config}")
done
[[ ${#CONFIGS[@]} -gt 0 ]] || { echo "no configs selected" >&2; exit 2; }

read -r -a GPU_ARR <<< "${GPUS:-0}"
[[ ${#GPU_ARR[@]} -gt 0 ]] || { echo "no GPUs selected" >&2; exit 2; }

declare -a BUCKETS
for index in "${!CONFIGS[@]}"; do
  bucket=$((index % ${#GPU_ARR[@]}))
  BUCKETS[$bucket]+="${CONFIGS[$index]}"$'\n'
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  for bucket in "${!GPU_ARR[@]}"; do
    gpu="${GPU_ARR[$bucket]}"
    while IFS= read -r config; do
      [[ -z "${config}" ]] && continue
      DRY_RUN=1 "${SCRIPT_DIR}/train_nm_tucker.sh" "${config}" --device "${gpu}" "$@"
    done <<< "${BUCKETS[$bucket]:-}"
  done
  exit 0
fi

stamp="$(date +%Y%m%d_%H%M%S)"
declare -a PIDS USED_GPUS
for bucket in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$bucket]}"
  configs="${BUCKETS[$bucket]:-}"
  [[ -z "${configs//[$'\n']/}" ]] && continue
  (
    worker_rc=0
    while IFS= read -r config; do
      [[ -z "${config}" ]] && continue
      name="${config%.yaml}"
      log="${LOG_DIR}/${name}_gpu${gpu}_${stamp}.log"
      echo "[gpu ${gpu}] launching ${name} -> ${log}" >&2
      if "${SCRIPT_DIR}/train_nm_tucker.sh" "${config}" --device "${gpu}" "$@" >"${log}" 2>&1; then
        echo "[gpu ${gpu}] OK ${name}" >&2
      else
        echo "[gpu ${gpu}] FAIL ${name} (see ${log})" >&2
        worker_rc=1
      fi
    done <<< "${configs}"
    exit "${worker_rc}"
  ) &
  PIDS+=("$!")
  USED_GPUS+=("${gpu}")
done

rc=0
for index in "${!PIDS[@]}"; do
  wait "${PIDS[$index]}" || {
    echo "worker on gpu ${USED_GPUS[$index]} failed" >&2
    rc=1
  }
done
exit "${rc}"
