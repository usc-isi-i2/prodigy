#!/usr/bin/env bash
# Evaluate terminal models using background context and held-out NM positives.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
MODEL_LIST="${MODEL_LIST:-${SCRIPT_DIR}/model_list.txt}"
[[ -f "${MODEL_LIST}" ]] || { echo "missing ${MODEL_LIST}; run make_model_list.py" >&2; exit 2; }
IFS=',' read -r -a GPU_ARR <<< "${GPUS:-0}"
for gpu in "${GPU_ARR[@]}"; do
  [[ "${gpu}" =~ ^[0-3]$ ]] || { echo "refusing GPU ${gpu}; only 0-3 are ours" >&2; exit 2; }
done
export PATH="/home/mhchu/miniconda3/bin:${PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"
python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
  --model-list "${MODEL_LIST}" \
  --data-root "${DATA_ROOT:-/dataMeR1/phil/data}" \
  --datasets ukr_rus_twitter,covid19_twitter,midterm,covid_political,election2020,ukr_rus_suspended,twibot20,cp_hk_twitter \
  --tasks nm --shots 3 --nm-n-way 30 --gpus "${GPUS:-0}" "$@" -- \
  --n_hop 2 \
  --neighbor_sampling_hop_sizes 9,9 \
  --neighbor_sampling_node_limit 101 \
  --neighbor_matching_walk_hops 1 \
  --edge_view static_background \
  --target_edge_view static_holdout \
  --neighbor_matching_edge_split True
