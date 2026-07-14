#!/usr/bin/env bash
# Train ALL 8 single-source NM configs across a pool of GPUs.
#
# There are 8 configs but usually fewer GPUs, so configs are partitioned
# round-robin across the GPU pool and each GPU runs its share SEQUENTIALLY (one
# background worker per GPU). With the default 4 GPUs that's 2 runs per GPU.
#
#   ./run_all_train_tucker.sh                    # GPUs 0 1 2 3
#   GPUS="0 1 2 3 4 5 6 7" ./run_all_train_tucker.sh   # one GPU per config (fastest)
#   GPUS="2 3" ./run_all_train_tucker.sh         # 2 GPUs -> 4 runs each
#   DRY_RUN=1 ./run_all_train_tucker.sh          # print commands, don't launch
#   SKIP="ukr_rus_twitter.yaml" ./run_all_train_tucker.sh   # skip configs (space-sep)
#
# Each run gets its own --device <gpu> and its own log under ./run_logs/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/run_logs"
mkdir -p "${LOG_DIR}"

ALL_CONFIGS=(
  ukr_rus_twitter.yaml
  covid19_twitter.yaml
  midterm.yaml
  covid_political.yaml
  election2020.yaml
  ukr_rus_suspended.yaml
  twibot20.yaml
  cp_hk_twitter.yaml
)

# Optional skip list (space-separated config filenames). Portable to bash 3.2
# (no associative arrays): match each config against the space-padded SKIP string.
is_skipped() { [[ " ${SKIP:-} " == *" $1 "* ]]; }
CONFIGS=()
for c in "${ALL_CONFIGS[@]}"; do
  if is_skipped "$c"; then echo "skipping ${c}" >&2; continue; fi
  CONFIGS+=("$c")
done

read -r -a GPU_ARR <<< "${GPUS:-0 1 2 3}"
[[ ${#GPU_ARR[@]} -lt 1 ]] && { echo "need at least one GPU" >&2; exit 2; }

stamp="$(date +%Y%m%d_%H%M%S)"

# Assign configs to GPU buckets round-robin: bucket[g] holds a newline list.
declare -a BUCKET
for i in "${!CONFIGS[@]}"; do
  g=$(( i % ${#GPU_ARR[@]} ))
  BUCKET[$g]+="${CONFIGS[$i]}"$'\n'
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  for g in "${!GPU_ARR[@]}"; do
    gpu="${GPU_ARR[$g]}"
    while IFS= read -r cfg; do
      [[ -z "$cfg" ]] && continue
      DRY_RUN=1 "${SCRIPT_DIR}/train_nm_tucker.sh" "${cfg}" --device "${gpu}" "$@"
    done <<< "${BUCKET[$g]:-}"
  done
  exit 0
fi

# One background worker per GPU; each processes its bucket sequentially.
declare -a PIDS GPUS_USED
for g in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$g]}"
  bucket="${BUCKET[$g]:-}"
  [[ -z "${bucket//[$'\n']/}" ]] && continue
  (
    while IFS= read -r cfg; do
      [[ -z "$cfg" ]] && continue
      name="${cfg%.yaml}"
      log="${LOG_DIR}/${name}_gpu${gpu}_${stamp}.log"
      echo "[gpu ${gpu}] launching ${name} -> ${log}" >&2
      if "${SCRIPT_DIR}/train_nm_tucker.sh" "${cfg}" --device "${gpu}" "$@" >"${log}" 2>&1; then
        echo "[gpu ${gpu}] OK   ${name}" >&2
      else
        echo "[gpu ${gpu}] FAIL ${name} (see ${log})" >&2
      fi
    done <<< "${bucket}"
  ) &
  PIDS+=("$!"); GPUS_USED+=("${gpu}")
done

rc=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}" || { echo "worker for gpu ${GPUS_USED[$i]} reported a failure" >&2; rc=1; }
done
echo "all workers done (rc=${rc}); per-run status is in ${LOG_DIR}/" >&2
exit "${rc}"
