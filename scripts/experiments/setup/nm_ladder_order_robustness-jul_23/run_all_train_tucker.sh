#!/usr/bin/env bash
# Train the order-robustness NM rungs across a pool of GPUs. Configs are partitioned
# round-robin across the pool; each GPU runs its share SEQUENTIALLY (one background
# worker per GPU). NM episodes are tiny (~2GB), so these coexist with other jobs.
#
#   GATE=1 ./run_all_train_tucker.sh             # gate run only (do this FIRST)
#   ./run_all_train_tucker.sh                    # the 11 rungs, GPUs 0 1 2 3
#   GPUS="0 1 2 3 4 5" ./run_all_train_tucker.sh # 6 GPUs -> 2 rungs each
#   DRY_RUN=1 ./run_all_train_tucker.sh          # print commands, don't launch
#   SKIP="train_ordC_r7.yaml" ./run_all_train_tucker.sh   # skip configs (space-sep)
#
# Each run gets its own --device <gpu> and its own log under ./run_logs/.
#
# ORDER OF OPERATIONS: run GATE=1 first and check it with check_gate.py. It re-trains a
# rung whose answer is already published (order A rung 4) through the source-subset
# shortcut. If the gate does not reproduce that row, the shortcut is invalid and all 11
# rungs below it are meaningless -- do not launch them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/run_logs"
mkdir -p "${LOG_DIR}"

if [[ "${GATE:-0}" == "1" ]]; then
  ALL_CONFIGS=(train_gate_ordA_r4.yaml)
else
  # 11 rungs: order B r3..r7, order C r2..r7 (the rest are reused -- see manifest.csv).
  ALL_CONFIGS=(
    train_ordB_r3.yaml
    train_ordB_r4.yaml
    train_ordB_r5.yaml
    train_ordB_r6.yaml
    train_ordB_r7.yaml
    train_ordC_r2.yaml
    train_ordC_r3.yaml
    train_ordC_r4.yaml
    train_ordC_r5.yaml
    train_ordC_r6.yaml
    train_ordC_r7.yaml
  )
fi

# Optional skip list (space-separated config filenames). Portable to bash 3.2.
is_skipped() { [[ " ${SKIP:-} " == *" $1 "* ]]; }
CONFIGS=()
for c in "${ALL_CONFIGS[@]}"; do
  if is_skipped "$c"; then echo "skipping ${c}" >&2; continue; fi
  if [[ ! -f "${SCRIPT_DIR}/${c}" ]]; then
    echo "MISSING config ${c} -- run make_configs.py first" >&2; exit 2
  fi
  CONFIGS+=("$c")
done

read -r -a GPU_ARR <<< "${GPUS:-0 1 2 3}"
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
