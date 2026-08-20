#!/usr/bin/env bash
# Evaluate one checkpoint on a requested target set using the locked fair-2-hop protocol.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
[[ $# -eq 4 ]] || {
  echo "usage: $0 <model label> <checkpoint> <datasets csv> <gpu>" >&2
  exit 2
}
MODEL_LABEL="$1"
CHECKPOINT="$2"
DATASETS="$3"
GPU="$4"
[[ "${GPU}" == "1" ]] || { echo "streaming evaluation is restricted to GPU 1" >&2; exit 2; }
[[ -s "${CHECKPOINT}" ]] || { echo "missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }

RUN_LOG_DIR="${SCRIPT_DIR}/run_logs"
mkdir -p "${RUN_LOG_DIR}"
MODEL_LIST="${RUN_LOG_DIR}/model_list_${MODEL_LABEL}.txt"
printf '%s %s\n' "${MODEL_LABEL}" "${CHECKPOINT}" >"${MODEL_LIST}"

export PATH="/home/mhchu/miniconda3/bin:${PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"
python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
  --model-list "${MODEL_LIST}" \
  --data-root "${DATA_ROOT:-/dataMeR1/phil/data}" \
  --datasets "${DATASETS}" \
  --tasks nm \
  --shots 3 \
  --nm-n-way 30 \
  --gpus "${GPU}" \
  --continue-on-error \
  -- \
  --n_hop 2 \
  --neighbor_sampling_hop_sizes 9,9 \
  --neighbor_sampling_node_limit 101 \
  --neighbor_matching_walk_hops 1
