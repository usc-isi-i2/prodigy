#!/usr/bin/env bash
# Train all three NM transfer-matrix sources IN PARALLEL, one per GPU.
#
#   ./run_all_train_tucker.sh                 # ukr->gpu0, covid->gpu1, merged->gpu2
#   GPUS="2 3 5" ./run_all_train_tucker.sh    # custom GPU assignment (same order)
#   DRY_RUN=1 ./run_all_train_tucker.sh       # print commands, don't launch
#
# Each run gets its own --device <gpu> (overrides the config's device) and its
# own stdout/stderr log under ./run_logs/. Extra args are forwarded to every run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/run_logs"
mkdir -p "${LOG_DIR}"

CONFIGS=(ukr_nm.yaml covid_nm.yaml merged_nm.yaml)
# GPU per config, same order. Override with GPUS="a b c".
read -r -a GPU_ARR <<< "${GPUS:-0 1 2}"
if [[ ${#GPU_ARR[@]} -lt ${#CONFIGS[@]} ]]; then
  echo "need ${#CONFIGS[@]} GPUs, got: '${GPUS:-0 1 2}'" >&2
  exit 2
fi

stamp="$(date +%Y%m%d_%H%M%S)"
declare -a PIDS NAMES LOGS

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  gpu="${GPU_ARR[$i]}"
  name="${cfg%.yaml}"
  log="${LOG_DIR}/${name}_gpu${gpu}_${stamp}.log"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    DRY_RUN=1 "${SCRIPT_DIR}/train_nm_tucker.sh" "${cfg}" --device "${gpu}" "$@"
    continue
  fi

  echo "launching ${name} on GPU ${gpu} -> ${log}" >&2
  "${SCRIPT_DIR}/train_nm_tucker.sh" "${cfg}" --device "${gpu}" "$@" \
    >"${log}" 2>&1 &
  PIDS+=("$!"); NAMES+=("${name}"); LOGS+=("${log}")
done

[[ "${DRY_RUN:-0}" == "1" ]] && exit 0

# Wait for all, report per-run exit status.
rc=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "OK   ${NAMES[$i]} (${LOGS[$i]})" >&2
  else
    code=$?
    echo "FAIL ${NAMES[$i]} exit=${code} (${LOGS[$i]})" >&2
    rc=1
  fi
done
exit "${rc}"
