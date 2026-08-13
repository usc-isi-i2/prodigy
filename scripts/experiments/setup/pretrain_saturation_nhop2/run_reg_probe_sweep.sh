#!/usr/bin/env bash
# Valid regression only: frozen encoder + fitted ridge probe, with n_hop=2 embeddings.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
MODEL_LIST="${MODEL_LIST:-${SCRIPT_DIR}/model_list.txt}"
[[ -s "${MODEL_LIST}" ]] || { echo "missing ${MODEL_LIST}" >&2; exit 2; }

export PATH="/home/mhchu/miniconda3/bin:${PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/dataMeR1/phil/data}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/scripts/experiments/analysis/transfer/ablations/prodigy_nm/saturation/pretrain_saturation_nhop2/data/reg_probe}"
GPU="${GPU:-0}"
DATASETS="${DATASETS:-midterm,twibot20,ukr_rus_twitter,covid19_twitter}"
mkdir -p "${OUT_DIR}"

graph_path_of() {
  case "$1" in
    ukr_rus_twitter) echo "ukr_rus_twitter/graphs/retweet_graph_parquet.pt" ;;
    covid19_twitter) echo "covid19_twitter/graphs/retweet_graph_parquet.pt" ;;
    midterm) echo "midterm/graphs/retweet_graph_parquet.pt" ;;
    twibot20) echo "twibot20/graphs/retweet_graph.pt" ;;
    *) return 1 ;;
  esac
}

IFS=',' read -r -a DATASET_ARR <<< "${DATASETS}"
for dataset in "${DATASET_ARR[@]}"; do
  rel="$(graph_path_of "${dataset}")" || { echo "unknown dataset: ${dataset}" >&2; exit 2; }
  cmd=(python3 scripts/eval/regression_probe_sweep.py
       --graph "${DATA_ROOT}/${rel}" --dataset "${dataset}"
       --model-list "${MODEL_LIST}" --out-dir "${OUT_DIR}"
       --targets followers_count,account_age_days --transform log1p
       --shots 10 --n-query 12 --episodes 500 --alpha 1.0
       --n-hop 2 --hop-sizes 9,9 --node-limit 101
       --batch-size "${ENCODE_BATCH_SIZE:-128}" --device "cuda:${GPU}")
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY:'; printf ' %q' "${cmd[@]}"; printf '\n'
  else
    "${cmd[@]}"
  fi
done
