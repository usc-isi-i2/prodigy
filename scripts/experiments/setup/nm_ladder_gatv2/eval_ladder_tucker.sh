#!/usr/bin/env bash
# Evaluate all eight GATv2 rungs on all eight NM graphs (64 paired jobs).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_LIST="${SCRIPT_DIR}/model_list.txt"
[[ -s "${MODEL_LIST}" ]] || { echo "missing ${MODEL_LIST}; run make_model_list.sh" >&2; exit 2; }
[[ "$(wc -l < "${MODEL_LIST}" | tr -d ' ')" == "8" ]] || {
  echo "${MODEL_LIST} must contain exactly 8 models" >&2
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
  --datasets ukr_rus_twitter,covid19_twitter,midterm,covid_political,election2020,ukr_rus_suspended,twibot20,cp_hk_twitter \
  --tasks nm \
  --shots 3 \
  --nm-n-way 30 \
  --gnn-type gat \
  --gpus "${GPUS:-}" \
  "$@"
