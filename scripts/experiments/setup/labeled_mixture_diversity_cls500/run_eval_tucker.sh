#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
mkdir -p "${SCRIPT_DIR}/run_logs"
read -r -a GPU_ARR <<< "${GPUS:-0 1}"
for gpu in "${GPU_ARR[@]}"; do
  [[ "${gpu}" =~ ^[01]$ ]] || { echo "only Tucker GPUs 0 and 1 are owned" >&2; exit 2; }
done
export PATH="/home/mhchu/miniconda3/bin:${PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"
python3 "${SCRIPT_DIR}/make_plan.py" --check
stamp="$(date +%Y%m%d_%H%M%S)"
pids=()
for index in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$index]}"
  python3 -u "${SCRIPT_DIR}/evaluate.py" --device "${gpu}" \
    --shard-index "${index}" --num-shards "${#GPU_ARR[@]}" \
    >"${SCRIPT_DIR}/run_logs/eval_shard${index}_gpu${gpu}_${stamp}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done
exit "${rc}"
