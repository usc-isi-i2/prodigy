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

# The target-only checkpoints are the existing singleton models. Only the shared
# all-five endpoint requires new training.
timeout --signal=TERM --kill-after=60 1800 \
  python3 -u "${SCRIPT_DIR}/run_train.py" --device 0 \
  --model-prefix labmix500_k5_all \
  >"${SCRIPT_DIR}/run_logs/control_train_all5.log" 2>&1

pids=()
for index in 0 1; do
  python3 -u "${SCRIPT_DIR}/evaluate.py" --device "${index}" \
    --shard-index "${index}" --num-shards 2 --mode controls \
    --results "${SCRIPT_DIR}/controls_seed0.jsonl" \
    >"${SCRIPT_DIR}/run_logs/control_eval_shard${index}_gpu${index}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done
exit "${rc}"
