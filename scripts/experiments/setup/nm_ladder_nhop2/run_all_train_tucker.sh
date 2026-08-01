#!/usr/bin/env bash
# Train one phase of the 2-hop NM ladder.
#
#   PHASE=A ./run_all_train_tucker.sh             # canonical order, 8 models
#   PHASE=robustness ./run_all_train_tucker.sh    # remaining B/C sets, 13 models
#   PHASE=all ./run_all_train_tucker.sh           # all 21 unique sets
#   PHASE=smoke ./run_all_train_tucker.sh         # 200-step election stress test
#   DRY_RUN=1 PHASE=A ./run_all_train_tucker.sh
#
# Defaults to one owned GPU because every worker loads the 104 GB all8 graph and 2-hop
# episode prefetch is substantially larger than 1-hop. Increase GPUS only after the
# smoke run and a fresh host-RAM/GPU check. Never use Tucker GPUs 4-7.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/run_logs"
mkdir -p "${LOG_DIR}"

PHASE="${PHASE:-A}"
case "${PHASE}" in
  A|robustness|all)
    ALL_CONFIGS=()
    while IFS= read -r config; do
      [[ -n "${config}" ]] && ALL_CONFIGS+=("${config}")
    done < <(python3 "${SCRIPT_DIR}/make_configs.py" --list-configs "${PHASE}")
    ;;
  smoke)
    ALL_CONFIGS=(configs/smoke_election.yaml)
    ;;
  *)
    echo "unknown PHASE=${PHASE}; expected smoke, A, robustness, or all" >&2
    exit 2
    ;;
esac

python3 "${SCRIPT_DIR}/make_configs.py" --check >/dev/null

is_skipped() { [[ " ${SKIP:-} " == *" $1 "* ]]; }
CONFIGS=()
for config in "${ALL_CONFIGS[@]}"; do
  if is_skipped "${config}"; then
    echo "skipping ${config}" >&2
    continue
  fi
  CONFIGS+=("${config}")
done
[[ ${#CONFIGS[@]} -gt 0 ]] || { echo "no configs selected" >&2; exit 2; }

read -r -a GPU_ARR <<< "${GPUS:-0}"
[[ ${#GPU_ARR[@]} -gt 0 ]] || { echo "need at least one GPU" >&2; exit 2; }
for gpu in "${GPU_ARR[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-3]$ ]]; then
    echo "refusing GPU ${gpu}: this project owns only Tucker GPUs 0-3" >&2
    exit 2
  fi
done

stamp="$(date +%Y%m%d_%H%M%S)"
declare -a BUCKET
for index in "${!CONFIGS[@]}"; do
  bucket=$((index % ${#GPU_ARR[@]}))
  BUCKET[$bucket]+="${CONFIGS[$index]}"$'\n'
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "phase=${PHASE} configs=${#CONFIGS[@]} gpus=${GPU_ARR[*]}" >&2
  for bucket in "${!GPU_ARR[@]}"; do
    gpu="${GPU_ARR[$bucket]}"
    while IFS= read -r config; do
      [[ -z "${config}" ]] && continue
      DRY_RUN=1 "${SCRIPT_DIR}/train_nm_tucker.sh" "${config}" --device "${gpu}" "$@"
    done <<< "${BUCKET[$bucket]:-}"
  done
  exit 0
fi

declare -a PIDS GPUS_USED
for bucket in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$bucket]}"
  configs="${BUCKET[$bucket]:-}"
  [[ -z "${configs//[$'\n']/}" ]] && continue
  (
    while IFS= read -r config; do
      [[ -z "${config}" ]] && continue
      name="$(basename "${config%.yaml}")"
      log="${LOG_DIR}/${name}_gpu${gpu}_${stamp}.log"
      echo "[gpu ${gpu}] launching ${name} -> ${log}" >&2
      if "${SCRIPT_DIR}/train_nm_tucker.sh" "${config}" --device "${gpu}" "$@" >"${log}" 2>&1; then
        echo "[gpu ${gpu}] OK   ${name}" >&2
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
    echo "worker for gpu ${GPUS_USED[$index]} reported a failure" >&2
    rc=1
  }
done
echo "phase=${PHASE} workers done (rc=${rc}); logs=${LOG_DIR}" >&2
exit "${rc}"
