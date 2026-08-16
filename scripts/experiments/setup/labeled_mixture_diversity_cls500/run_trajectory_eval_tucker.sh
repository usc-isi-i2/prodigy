#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
mkdir -p "${SCRIPT_DIR}/run_logs"
export PATH="/home/mhchu/miniconda3/bin:${PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"

read -r -a GPU_ARR <<< "${EVAL_GPUS:-0 1}"
for gpu in "${GPU_ARR[@]}"; do
  [[ "${gpu}" =~ ^[01]$ ]] || { echo "only Tucker GPUs 0 and 1 are owned" >&2; exit 2; }
done
num_shards="${#GPU_ARR[@]}"
state_root="${CONTINUATION_STATE_ROOT:-${REPO_ROOT}/state_labmix500_continuation}"

for spec in 750:250 1000:500; do
  global_step="${spec%%:*}"
  local_step="${spec##*:}"
  for mode in heldout controls; do
    stamp="$(date +%Y%m%d_%H%M%S)"
    pids=()
    for index in "${!GPU_ARR[@]}"; do
      gpu="${GPU_ARR[$index]}"
      python3 -u "${SCRIPT_DIR}/evaluate.py" --device "${gpu}" \
        --shard-index "${index}" --num-shards "${num_shards}" --mode "${mode}" \
        --state-root "${state_root}" \
        --checkpoint-prefix labmixcont --checkpoint-step "${local_step}" \
        --training-steps "${global_step}" --run-stamp "seed0_step${global_step}" \
        --results "${SCRIPT_DIR}/trajectory_step${global_step}_${mode}.jsonl" \
        >"${SCRIPT_DIR}/run_logs/trajectory_${mode}_step${global_step}_shard${index}_gpu${gpu}_${stamp}.log" 2>&1 &
      pids+=("$!")
    done
    rc=0
    for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done
    (( rc == 0 )) || exit "${rc}"
  done
done
