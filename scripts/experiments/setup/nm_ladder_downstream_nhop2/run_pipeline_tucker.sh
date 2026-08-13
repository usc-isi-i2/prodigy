#!/usr/bin/env bash
# Resolve -> smoke -> 4-GPU classification -> 4-worker static LP -> assemble.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ANALYSIS_DIR="${REPO_ROOT}/scripts/experiments/analysis/transfer/ablations/downstream/two_hop/nm_ladder_downstream_nhop2"
RAW_ROOT="${RAW_ROOT:-${ANALYSIS_DIR}/data/raw}"
PAIR_DIR="${RAW_ROOT}/pair_lp"
RUNNER_OUT="${RAW_ROOT}/runner"
LOG_DIR="${SCRIPT_DIR}/run_logs"
STATUS_FILE="${LOG_DIR}/pipeline_status.txt"
PIPE_LOG="${LOG_DIR}/pipeline.log"
GPUS="${GPUS:-0,1,2,3}"
ONLY="${ONLY:-}"
PY="${PY:-/home/mhchu/miniconda3/envs/prodigy/bin/python}"
mkdir -p "${LOG_DIR}"

say() { echo "[$(date +%F_%T)] $*" | tee -a "${PIPE_LOG}"; }
status() {
  echo "PHASE=$1 STATUS=$2 TS=$(date +%F_%T) NOTE=${3:-}" > "${STATUS_FILE}"
  say "PHASE=$1 STATUS=$2 ${3:-}"
}
fail() { status "$1" FAILED "${2:-}"; exit 1; }
want() { [[ -z "${ONLY}" || "${ONLY}" == "$1" ]]; }

say "pipeline start: GPUs=${GPUS} raw_root=${RAW_ROOT}"

if want resolve; then
  status resolve RUNNING
  "${PY}" "${SCRIPT_DIR}/make_model_list.py" >> "${PIPE_LOG}" 2>&1 \
    || fail resolve "one or more completed ladder checkpoints could not be resolved"
  [[ "$(grep -cvE '^\s*(#|$)' "${SCRIPT_DIR}/model_list.txt")" == "39" ]] \
    || fail resolve "model list is not 39 physical encoders"
  status resolve OK "40 logical rows / 39 encoders"
fi

if want smoke && [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  status smoke RUNNING
  head -1 "${SCRIPT_DIR}/model_list.txt" > "${LOG_DIR}/model_list_smoke.txt"
  SMOKE_DIR="${LOG_DIR}/smoke_pair_lp"
  EXPECTED_MODELS=1 MODEL_LIST="${LOG_DIR}/model_list_smoke.txt" \
  OUT_DIR="${SMOKE_DIR}" DATASETS=cp_hk_twitter GPU="${GPUS%%,*}" \
    bash "${SCRIPT_DIR}/run_pair_lp_worker.sh" >> "${PIPE_LOG}" 2>&1 \
    || fail smoke "pair evaluator failed on cp_hk_twitter"
  "${PY}" "${SCRIPT_DIR}/check_pair_results.py" \
    --pair-dir "${SMOKE_DIR}" --model-list "${LOG_DIR}/model_list_smoke.txt" \
    --datasets cp_hk_twitter >> "${PIPE_LOG}" 2>&1 \
    || fail smoke "pair evaluator validity gate failed"
  status smoke OK
fi

if want classification; then
  status classification RUNNING
  MODEL_LIST="${SCRIPT_DIR}/model_list.txt" GPUS="${GPUS}" \
    bash "${SCRIPT_DIR}/run_classification_sweep.sh" >> "${PIPE_LOG}" 2>&1 \
    || fail classification "one or more classification jobs failed"
  export PATH="/home/mhchu/miniconda3/bin:${PATH}"
  source "$(conda info --base)/etc/profile.d/conda.sh" \
    || fail classification "could not activate conda"
  conda activate prodigy || fail classification "could not activate prodigy"
  "${PY}" "${REPO_ROOT}/scripts/harness/benchmark_tasks/parse_benchmark_eval_logs.py" \
    --log-root "${REPO_ROOT}/log" --out-dir "${RUNNER_OUT}" --overwrite \
    >> "${PIPE_LOG}" 2>&1 \
    || fail classification "classification log parse failed"
  status classification OK "156 jobs across GPUs ${GPUS}"
fi

if want static_lp; then
  status static_lp RUNNING
  MODEL_LIST="${SCRIPT_DIR}/model_list.txt" OUT_DIR="${PAIR_DIR}" GPUS="${GPUS}" \
  WORKER_LOG_DIR="${LOG_DIR}/pair_workers" \
    bash "${SCRIPT_DIR}/run_pair_lp_parallel.sh" >> "${PIPE_LOG}" 2>&1 \
    || fail static_lp "one or more graph workers failed"
  "${PY}" "${SCRIPT_DIR}/check_pair_results.py" \
    --pair-dir "${PAIR_DIR}" --model-list "${SCRIPT_DIR}/model_list.txt" \
    >> "${PIPE_LOG}" 2>&1 \
    || fail static_lp "195-cell completeness/validity gate failed"
  status static_lp OK "39 encoders x five shared pair sets on GPUs ${GPUS}"
fi

if want assemble; then
  status assemble RUNNING
  "${PY}" "${ANALYSIS_DIR}/assemble_results.py" \
    --row-map "${SCRIPT_DIR}/row_map.csv" \
    --classification-csv "${RUNNER_OUT}/node_classification/data/node_classification.csv" \
    --pair-dir "${PAIR_DIR}" --out-dir "${ANALYSIS_DIR}/data" \
    >> "${PIPE_LOG}" 2>&1 \
    || fail assemble "result matrix is incomplete"
  status assemble OK
fi

status pipeline OK
