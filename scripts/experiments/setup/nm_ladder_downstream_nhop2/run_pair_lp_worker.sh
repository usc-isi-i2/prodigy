#!/usr/bin/env bash
# Score one or more static-LP graphs serially on one GPU. Multiple workers run in parallel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
MODEL_LIST="${MODEL_LIST:-${SCRIPT_DIR}/model_list.txt}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/scripts/experiments/analysis/nm_ladder_downstream_nhop2/data/raw/pair_lp}"
DATA_ROOT="${DATA_ROOT:-/dataMeR1/phil/data}"
GPU="${GPU:-0}"
DATASETS="${DATASETS:-cp_hk_twitter}"

[[ "${GPU}" =~ ^[0-3]$ ]] || { echo "refusing GPU ${GPU}; only 0-3 are ours" >&2; exit 2; }
[[ -s "${MODEL_LIST}" ]] || { echo "missing ${MODEL_LIST}" >&2; exit 2; }
COUNT="$(grep -cvE '^\s*(#|$)' "${MODEL_LIST}")"
[[ "${COUNT}" == "${EXPECTED_MODELS:-39}" ]] || {
  echo "${MODEL_LIST} has ${COUNT} models; expected ${EXPECTED_MODELS:-39}" >&2
  exit 2
}

graph_path_of() {
  case "$1" in
    ukr_rus_twitter) echo "ukr_rus_twitter/graphs/retweet_graph_parquet.pt" ;;
    covid19_twitter) echo "covid19_twitter/graphs/retweet_graph_parquet.pt" ;;
    midterm) echo "midterm/graphs/retweet_graph_parquet.pt" ;;
    twibot20) echo "twibot20/graphs/retweet_graph.pt" ;;
    cp_hk_twitter) echo "cp_hk_twitter/graphs/retweet_graph.pt" ;;
    *) return 1 ;;
  esac
}

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  export PATH="/home/mhchu/miniconda3/bin:${PATH}"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate prodigy
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  mkdir -p "${OUT_DIR}"
fi
cd "${REPO_ROOT}"

IFS=',' read -r -a DATASET_ARRAY <<< "${DATASETS}"
for dataset in "${DATASET_ARRAY[@]}"; do
  relative="$(graph_path_of "${dataset}")" || {
    echo "unknown static-LP dataset: ${dataset}" >&2
    exit 2
  }
  CMD=(python3 scripts/eval/pair_link_sweep.py
    --graph "${DATA_ROOT}/${relative}"
    --dataset "${dataset}"
    --model-list "${MODEL_LIST}"
    --out-dir "${OUT_DIR}"
    --negative-kinds "${NEGATIVE_KINDS:-degree_matched,random,hard_2hop}"
    --n-hop 2
    --hop-sizes 9,9
    --node-limit 101
    --batch-size "${BATCH_SIZE:-256}"
    --resume
    --device "cuda:${GPU}")
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY_RUN gpu=%s dataset=%s: ' "${GPU}" "${dataset}"
    printf '%q ' "${CMD[@]}"
    printf '\n'
  else
    echo "=== static LP ${dataset} on GPU ${GPU} ==="
    "${CMD[@]}"
  fi
done
