#!/usr/bin/env bash
# 19 checkpoints x 4 labelled graphs, fixed 10-shot episodes, compute-matched h2 eval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
MODEL_LIST="${MODEL_LIST:-${SCRIPT_DIR}/model_list.txt}"
[[ -s "${MODEL_LIST}" ]] || { echo "missing ${MODEL_LIST}" >&2; exit 2; }
[[ "$(grep -cvE '^\s*(#|$)' "${MODEL_LIST}")" == "19" ]] || {
  echo "${MODEL_LIST} must contain exactly 19 checkpoints" >&2
  exit 2
}

export PATH="/home/mhchu/miniconda3/bin:${PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"

python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
  --model-list "${MODEL_LIST}" \
  --data-root "${DATA_ROOT:-/dataMeR1/phil/data}" \
  --datasets covid_political,election2020,ukr_rus_suspended,twibot20 \
  --tasks pl --shots 10 --workers 2 --continue-on-error \
  --gpus "${GPUS:-}" \
  "$@" -- --n_hop 2 --neighbor_sampling_hop_sizes 9,9 \
  --neighbor_sampling_node_limit 101 --neighbor_matching_walk_hops 1
