#!/usr/bin/env bash
# Build CP-HK graph artifacts and train NM on Tucker without Slurm.
set -euo pipefail

CONDA_SH="${CONDA_SH:-/home/mhchu/miniconda3/etc/profile.d/conda.sh}"
REPO_ROOT="${REPO_ROOT:-/dataMeR1/phil/gfm/prodigy}"
DATA_ROOT="${DATA_ROOT:-/dataMeR1/phil/data/cp_hk_twitter}"
LOG_ROOT="${LOG_ROOT:-/dataMeR1/phil/logs}"
GPU_ID="${GPU_ID:-1}"
BUILD_ONLY="${BUILD_ONLY:-0}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"
MAX_RECORDS="${MAX_RECORDS:-0}"
CP_HK_RAW_FILES="${CP_HK_RAW_FILES:-an_cp-hk.twitter.v7-ground-truth.2020-04-07_2020-08-23.json.gz an_cp-hk.twitter.v7-ground-truth.2020-08-24_2020-09-13.json.gz}"

RAW_DIR="${DATA_ROOT}/raw"
PARQUET_DIR="${DATA_ROOT}/parquet"
EMB_DIR="${DATA_ROOT}/embeddings"
BIO_ROOT="${DATA_ROOT}/bio_embeddings/gte-multilingual-base/version=v001"
GRAPH_DIR="${DATA_ROOT}/graphs"
EMB_PATH="${EMB_DIR}/user_bio_embeddings_gte_multilingual_base.pt"
GRAPH_PATH="${GRAPH_DIR}/retweet_graph.pt"
CONFIG_PATH="${REPO_ROOT}/scripts/experiments/cp_hk_twitter/cp_hk_nm.yaml"

mkdir -p "${LOG_ROOT}" "${RAW_DIR}" "${PARQUET_DIR}" "${EMB_DIR}" "${BIO_ROOT}" "${GRAPH_DIR}"
cd "${REPO_ROOT}"

source "${CONDA_SH}"

if [[ "${TRAIN_ONLY}" != "1" ]]; then
  conda activate tweet-embeddings-v001
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"

  read -r -a raw_files <<< "${CP_HK_RAW_FILES}"
  inputs=()
  for raw_file in "${raw_files[@]}"; do
    inputs+=("${RAW_DIR}/${raw_file}")
  done
  for input in "${inputs[@]}"; do
    if [[ ! -s "${input}" ]]; then
      echo "Missing raw input: ${input}" >&2
      exit 1
    fi
  done

  convert_args=()
  for input in "${inputs[@]}"; do
    convert_args+=(--input "${input}")
  done
  python -u scripts/graph_construction/cp_hk_json_to_parquet.py \
    "${convert_args[@]}" \
    --out-dir "${PARQUET_DIR}" \
    --max-records "${MAX_RECORDS}"

  python -u scripts/graph_construction/build_cp_hk_bio_embeddings.py \
    --users-parquet "${PARQUET_DIR}/user_bios.parquet" \
    --out "${EMB_PATH}" \
    --bio-output-root "${BIO_ROOT}" \
    --batch-size 1024

  python -u scripts/graph_construction/generate_cp_hk_retweet_graph_from_parquet.py \
    --parquet-dir "${PARQUET_DIR}" \
    --embeddings "${EMB_PATH}" \
    --out "${GRAPH_PATH}"
fi

if [[ "${BUILD_ONLY}" == "1" ]]; then
  echo "BUILD_ONLY=1; stopping before training."
  exit 0
fi

conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python -u experiments/run_single_experiment.py --config "${CONFIG_PATH}"
