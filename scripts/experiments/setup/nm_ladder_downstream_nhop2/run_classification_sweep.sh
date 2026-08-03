#!/usr/bin/env bash
# 39 fair-two-hop encoders x 4 labeled graphs, distributed over all four owned GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
MODEL_LIST="${MODEL_LIST:-${SCRIPT_DIR}/model_list.txt}"
GPUS="${GPUS:-0,1,2,3}"

[[ -s "${MODEL_LIST}" ]] || { echo "missing ${MODEL_LIST}; run make_model_list.py" >&2; exit 2; }
COUNT="$(grep -cvE '^\s*(#|$)' "${MODEL_LIST}")"
[[ "${COUNT}" == "${EXPECTED_MODELS:-39}" ]] || {
  echo "${MODEL_LIST} has ${COUNT} models; expected ${EXPECTED_MODELS:-39}" >&2
  exit 2
}
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
for gpu in "${GPU_ARRAY[@]}"; do
  [[ "${gpu}" =~ ^[0-3]$ ]] || { echo "refusing GPU ${gpu}; only 0-3 are ours" >&2; exit 2; }
done

CMD=(python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py
  --model-list "${MODEL_LIST}"
  --data-root "${DATA_ROOT:-/dataMeR1/phil/data}"
  --datasets covid_political,election2020,ukr_rus_suspended,twibot20
  --tasks classification
  --shots 10
  --workers "${WORKERS:-2}"
  --continue-on-error
  --gpus "${GPUS}"
  --
  --n_hop 2
  --neighbor_sampling_hop_sizes 9,9
  --neighbor_sampling_node_limit 101
  --neighbor_matching_walk_hops 1)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY_RUN: '
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

export PATH="/home/mhchu/miniconda3/bin:${PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"
"${CMD[@]}"
