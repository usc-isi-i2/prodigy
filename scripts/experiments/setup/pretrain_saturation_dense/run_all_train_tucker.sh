#!/usr/bin/env bash
# Train the 3 dense saturation arms across a pool of GPUs. Configs are partitioned
# round-robin; each GPU runs its share SEQUENTIALLY (one background worker per GPU).
#
#   ./run_all_train_tucker.sh                    # SERIAL on GPU 0 (default -- see below)
#   GPUS="0 1" ./run_all_train_tucker.sh         # 2 GPUs; splits all8 and covid apart
#   DRY_RUN=1 ./run_all_train_tucker.sh          # print commands, don't launch
#   SKIP="train_covid_dense.yaml" ./run_all_train_tucker.sh
#
# Each run is only 2100 steps. Measured from the historical runs' checkpoint mtimes:
# all8 ~7.6 steps/s (~5 min), ukr/covid ~4.5 steps/s (~8 min). LOADING dominates the wall
# clock, not training: the graphs are 104 GB (all8), 73 GB (covid) and 35 GB (ukr) on disk
# and expand several-fold in host RAM.
#
# Hence the SERIAL default. Checked 2026-07-27: Tucker has 1511 GB of RAM but only ~820 GB
# available and a load average near 180, so holding two of these graphs at once is a real
# risk of thrashing a shared box for very little gain -- the arms are minutes of compute
# and would contend on disk anyway. Parallelism is one env var away when the box is quiet.
#
# Config order is deliberate: with GPUS="0 1" the round-robin puts all8+ukr on the first
# GPU and covid on the second, which splits the two heaviest loads.
#
# Each run gets its own --device <gpu> and its own log under ./run_logs/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/run_logs"
mkdir -p "${LOG_DIR}"

ALL_CONFIGS=(
  train_all8_dense.yaml
  train_covid_dense.yaml
  train_ukr_dense.yaml
)

# Optional skip list (space-separated config filenames). Portable to bash 3.2.
is_skipped() { [[ " ${SKIP:-} " == *" $1 "* ]]; }
CONFIGS=()
for c in "${ALL_CONFIGS[@]}"; do
  if is_skipped "$c"; then echo "skipping ${c}" >&2; continue; fi
  CONFIGS+=("$c")
done

read -r -a GPU_ARR <<< "${GPUS:-0}"
[[ ${#GPU_ARR[@]} -lt 1 ]] && { echo "need at least one GPU" >&2; exit 2; }

stamp="$(date +%Y%m%d_%H%M%S)"

# Assign configs to GPU buckets round-robin.
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
