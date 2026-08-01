#!/usr/bin/env bash
# Run all eight GATv2 ladder rungs, one sequential bucket per selected GPU.
# Examples:
#   DRY_RUN=1 GPUS="0 1 2 3" ./run_all_train_tucker.sh
#   GPUS="0 1 2 3" ./run_all_train_tucker.sh
#   SKIP="train_1src.yaml train_8src.yaml" GPUS="0 1" ./run_all_train_tucker.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/run_logs"
ALL_CONFIGS=(
  train_1src.yaml train_2src.yaml train_3src.yaml train_4src.yaml
  train_5src.yaml train_6src.yaml train_7src.yaml train_8src.yaml
)

python3 "${SCRIPT_DIR}/validate_configs.py"

is_skipped() { [[ " ${SKIP:-} " == *" $1 "* ]]; }
CONFIGS=()
for config in "${ALL_CONFIGS[@]}"; do
  if is_skipped "${config}"; then
    echo "skipping ${config}" >&2
  else
    CONFIGS+=("${config}")
  fi
done

read -r -a GPU_ARR <<< "${GPUS:-0 1 2 3}"
[[ ${#GPU_ARR[@]} -gt 0 ]] || { echo "need at least one GPU" >&2; exit 2; }

declare -a BUCKET
for index in "${!CONFIGS[@]}"; do
  bucket=$(( index % ${#GPU_ARR[@]} ))
  BUCKET[$bucket]+="${CONFIGS[$index]}"$'\n'
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  for bucket in "${!GPU_ARR[@]}"; do
    gpu="${GPU_ARR[$bucket]}"
    while IFS= read -r config; do
      [[ -n "${config}" ]] || continue
      DRY_RUN=1 "${SCRIPT_DIR}/train_nm_tucker.sh" "${config}" --device "${gpu}" "$@"
    done <<< "${BUCKET[$bucket]:-}"
  done
  exit 0
fi

mkdir -p "${LOG_DIR}"
stamp="$(date +%Y%m%d_%H%M%S)"
declare -a PIDS GPUS_USED
for bucket in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$bucket]}"
  configs="${BUCKET[$bucket]:-}"
  [[ -n "${configs//$'\n'/}" ]] || continue
  (
    while IFS= read -r config; do
      [[ -n "${config}" ]] || continue
      name="${config%.yaml}"
      log="${LOG_DIR}/${name}_gpu${gpu}_${stamp}.log"
      echo "[gpu ${gpu}] launching ${name} -> ${log}" >&2
      if "${SCRIPT_DIR}/train_nm_tucker.sh" "${config}" --device "${gpu}" "$@" >"${log}" 2>&1; then
        echo "[gpu ${gpu}] OK ${name}" >&2
      else
        echo "[gpu ${gpu}] FAIL ${name} (see ${log})" >&2
        exit 1
      fi
    done <<< "${configs}"
  ) &
  PIDS+=("$!")
  GPUS_USED+=("${gpu}")
done

rc=0
for index in "${!PIDS[@]}"; do
  wait "${PIDS[$index]}" || {
    echo "worker for gpu ${GPUS_USED[$index]} failed" >&2
    rc=1
  }
done
exit "${rc}"
