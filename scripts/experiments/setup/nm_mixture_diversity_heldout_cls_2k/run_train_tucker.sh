#!/usr/bin/env bash
# Shard the exhaustive sweep across one or both currently owned Tucker GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
mkdir -p "${SCRIPT_DIR}/run_logs"

read -r -a GPU_ARR <<< "${GPUS:-0}"
[[ ${#GPU_ARR[@]} -gt 0 ]] || { echo "need at least one GPU" >&2; exit 2; }
for gpu in "${GPU_ARR[@]}"; do
  [[ "${gpu}" =~ ^[01]$ ]] || {
    echo "refusing GPU ${gpu}: this project currently owns only Tucker GPUs 0 and 1" >&2
    exit 2
  }
done

export PATH="/home/mhchu/miniconda3/bin:${PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"
python3 "${SCRIPT_DIR}/make_plan.py" --check

stamp="$(date +%Y%m%d_%H%M%S)"
declare -a PIDS
for index in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$index]}"
  cmd=(python3 -u "${SCRIPT_DIR}/run_sweep.py"
       --device "${gpu}" --shard-index "${index}" --num-shards "${#GPU_ARR[@]}"
       --targets "${TARGETS:-}" --sizes "${SIZES:-1,2,3,4}")
  [[ -z "${LIMIT:-}" ]] || cmd+=(--limit "${LIMIT}")
  [[ "${DRY_RUN:-0}" != "1" ]] || cmd+=(--dry-run)
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    "${cmd[@]}"
  else
    "${cmd[@]}" >"${SCRIPT_DIR}/run_logs/shard${index}_gpu${gpu}_${stamp}.log" 2>&1 &
    PIDS+=("$!")
  fi
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi
rc=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || rc=1
done
exit "${rc}"
