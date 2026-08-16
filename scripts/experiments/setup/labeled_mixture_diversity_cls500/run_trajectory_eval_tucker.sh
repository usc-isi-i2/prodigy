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

for spec in 750:250 1000:500; do
  global_step="${spec%%:*}"
  local_step="${spec##*:}"
  for mode in heldout controls; do
    stamp="$(date +%Y%m%d_%H%M%S)"
    pids=()
    for index in 0 1; do
      python3 -u "${SCRIPT_DIR}/evaluate.py" --device "${index}" \
        --shard-index "${index}" --num-shards 2 --mode "${mode}" \
        --state-root "${REPO_ROOT}/state_labmix500_continuation" \
        --checkpoint-prefix labmixcont --checkpoint-step "${local_step}" \
        --training-steps "${global_step}" --run-stamp "seed0_step${global_step}" \
        --results "${SCRIPT_DIR}/trajectory_step${global_step}_${mode}.jsonl" \
        >"${SCRIPT_DIR}/run_logs/trajectory_${mode}_step${global_step}_shard${index}_gpu${index}_${stamp}.log" 2>&1 &
      pids+=("$!")
    done
    rc=0
    for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done
    (( rc == 0 )) || exit "${rc}"
  done
done
