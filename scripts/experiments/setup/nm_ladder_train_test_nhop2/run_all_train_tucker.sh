#!/usr/bin/env bash
# Run the eight canonical rungs in parallel buckets, one process per owned GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/run_logs"
mkdir -p "${LOG_DIR}"

case "${PHASE:-all}" in
  all)
    ALL_CONFIGS=()
    while IFS= read -r config; do
      [[ -n "${config}" ]] && ALL_CONFIGS+=("${config}")
    done < <(python3 "${SCRIPT_DIR}/make_configs.py" --list-configs)
    ;;
  smoke) ALL_CONFIGS=(configs/smoke_election.yaml) ;;
  *) echo "unknown PHASE=${PHASE}; expected smoke or all" >&2; exit 2 ;;
esac
python3 "${SCRIPT_DIR}/make_configs.py" --check >/dev/null

is_skipped() { [[ " ${SKIP:-} " == *" $1 "* ]]; }
CONFIGS=()
for config in "${ALL_CONFIGS[@]}"; do
  is_skipped "${config}" || CONFIGS+=("${config}")
done
[[ ${#CONFIGS[@]} -gt 0 ]] || { echo "no configs selected" >&2; exit 2; }

read -r -a GPU_ARR <<< "${GPUS:-0}"
for gpu in "${GPU_ARR[@]}"; do
  [[ "${gpu}" =~ ^[0-3]$ ]] || { echo "refusing GPU ${gpu}; only 0-3 are ours" >&2; exit 2; }
done

declare -a BUCKET
for index in "${!CONFIGS[@]}"; do
  slot=$((index % ${#GPU_ARR[@]}))
  BUCKET[$slot]+="${CONFIGS[$index]}"$'\n'
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "phase=${PHASE:-all} configs=${#CONFIGS[@]} gpus=${GPU_ARR[*]}" >&2
  for slot in "${!GPU_ARR[@]}"; do
    while IFS= read -r config; do
      [[ -n "${config}" ]] || continue
      DRY_RUN=1 "${SCRIPT_DIR}/train_nm_tucker.sh" "${config}" --device "${GPU_ARR[$slot]}" "$@"
    done <<< "${BUCKET[$slot]:-}"
  done
  exit 0
fi

stamp="$(date +%Y%m%d_%H%M%S)"
declare -a PIDS USED
for slot in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$slot]}"; bucket="${BUCKET[$slot]:-}"
  [[ -n "${bucket//$'\n'/}" ]] || continue
  (
    while IFS= read -r config; do
      [[ -n "${config}" ]] || continue
      name="$(basename "${config%.yaml}")"
      log="${LOG_DIR}/${name}_gpu${gpu}_${stamp}.log"
      echo "[gpu ${gpu}] ${name} -> ${log}" >&2
      "${SCRIPT_DIR}/train_nm_tucker.sh" "${config}" --device "${gpu}" "$@" >"${log}" 2>&1
    done <<< "${bucket}"
  ) &
  PIDS+=("$!"); USED+=("${gpu}")
done
rc=0
for index in "${!PIDS[@]}"; do
  wait "${PIDS[$index]}" || { echo "GPU ${USED[$index]} worker failed" >&2; rc=1; }
done
exit "${rc}"
