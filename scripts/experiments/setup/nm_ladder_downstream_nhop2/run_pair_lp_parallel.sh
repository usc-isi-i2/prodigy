#!/usr/bin/env bash
# One to four concurrent graph workers, balanced around the two largest artifacts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS="${GPUS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
GPU_COUNT="${#GPU_ARRAY[@]}"
[[ "${GPU_COUNT}" -ge 1 && "${GPU_COUNT}" -le 4 ]] || {
  echo "GPUS must name one to four devices from 0-3" >&2
  exit 2
}
for gpu in "${GPU_ARRAY[@]}"; do
  [[ "${gpu}" =~ ^[0-3]$ ]] || { echo "refusing GPU ${gpu}; only 0-3 are ours" >&2; exit 2; }
done

LOG_DIR="${WORKER_LOG_DIR:-${SCRIPT_DIR}/run_logs/pair_workers}"
mkdir -p "${LOG_DIR}"
case "${GPU_COUNT}" in
  1) ASSIGNMENTS=("covid19_twitter,ukr_rus_twitter,twibot20,midterm,cp_hk_twitter") ;;
  2) ASSIGNMENTS=("covid19_twitter,midterm,cp_hk_twitter" "ukr_rus_twitter,twibot20") ;;
  3) ASSIGNMENTS=("covid19_twitter" "ukr_rus_twitter" "twibot20,midterm,cp_hk_twitter") ;;
  4) ASSIGNMENTS=("covid19_twitter" "ukr_rus_twitter" "midterm,cp_hk_twitter" "twibot20") ;;
esac

pids=()
for ((index=0; index<GPU_COUNT; index++)); do
  gpu="${GPU_ARRAY[$index]}"
  datasets="${ASSIGNMENTS[$index]}"
  echo "worker ${index}: GPU ${gpu} -> ${datasets}"
  GPU="${gpu}" DATASETS="${datasets}" \
    bash "${SCRIPT_DIR}/run_pair_lp_worker.sh" \
    > "${LOG_DIR}/worker_${index}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for ((index=0; index<GPU_COUNT; index++)); do
  if wait "${pids[$index]}"; then
    echo "worker ${index}: OK"
  else
    echo "worker ${index}: FAILED (see ${LOG_DIR}/worker_${index}.log)" >&2
    failed=1
  fi
done
exit "${failed}"
