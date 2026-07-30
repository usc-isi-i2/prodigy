#!/usr/bin/env bash
# Retrain the five missing matched-40k single-source NM checkpoints on Tucker.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs"
LOG_DIR="${SCRIPT_DIR}/run_logs/train"
mkdir -p "${LOG_DIR}"

CONFIGS=(
  midterm.yaml
  covid_political.yaml
  ukr_rus_suspended.yaml
  twibot20.yaml
  cp_hk_twitter.yaml
)
read -r -a GPU_ARR <<< "${TRAIN_GPUS:-0 1}"
[[ ${#GPU_ARR[@]} -gt 0 ]] || { echo "TRAIN_GPUS is empty" >&2; exit 2; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"

checkpoint_for() {
  local prefix="$1"
  find "${REPO_ROOT}/state" -maxdepth 3 -type f \
    -path "*/${prefix}_*/checkpoint/state_dict_40000.ckpt" \
    -print -quit 2>/dev/null
}

declare -a BUCKET
for i in "${!CONFIGS[@]}"; do
  bucket=$((i % ${#GPU_ARR[@]}))
  BUCKET[$bucket]+="${CONFIGS[$i]}"$'\n'
done

stamp="$(date +%Y%m%d_%H%M%S)"
declare -a PIDS GPUS_USED
for bucket in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$bucket]}"
  entries="${BUCKET[$bucket]:-}"
  [[ -n "${entries//[$'\n']/}" ]] || continue
  (
    worker_rc=0
    while IFS= read -r config_name; do
      [[ -n "${config_name}" ]] || continue
      config="${CONFIG_DIR}/${config_name}"
      prefix="$(awk '$1 == "prefix:" {print $2}' "${config}")"
      existing="$(checkpoint_for "${prefix}")"
      if [[ -n "${existing}" ]]; then
        echo "[gpu ${gpu}] reuse ${prefix}: ${existing}"
        continue
      fi
      log="${LOG_DIR}/${prefix}_gpu${gpu}_${stamp}.log"
      cmd=(python3 experiments/run_single_experiment.py
           --config "${config}" --device "${gpu}")
      if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf 'DRY_RUN:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        continue
      fi
      echo "[gpu ${gpu}] launch ${prefix} -> ${log}"
      if "${cmd[@]}" >"${log}" 2>&1; then
        echo "[gpu ${gpu}] OK ${prefix}"
      else
        echo "[gpu ${gpu}] FAILED ${prefix}; see ${log}" >&2
        worker_rc=1
      fi
    done <<< "${entries}"
    exit "${worker_rc}"
  ) &
  PIDS+=("$!")
  GPUS_USED+=("${gpu}")
done

rc=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}" || {
    echo "training worker on GPU ${GPUS_USED[$i]} failed" >&2
    rc=1
  }
done
exit "${rc}"
