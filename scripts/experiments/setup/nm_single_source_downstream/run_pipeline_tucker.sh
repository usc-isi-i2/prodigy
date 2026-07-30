#!/usr/bin/env bash
# train -> resolve -> evaluate -> assemble -> plot, designed for detached Tucker tmux.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ANALYSIS_DIR="${REPO_ROOT}/scripts/experiments/analysis/nm_single_source_downstream"
LOG_DIR="${SCRIPT_DIR}/run_logs"
mkdir -p "${LOG_DIR}"

STATUS_FILE="${LOG_DIR}/pipeline_status.txt"
PIPELOG="${LOG_DIR}/pipeline.log"
PY="${PY:-/home/mhchu/miniconda3/envs/prodigy/bin/python}"
TRAIN_GPUS="${TRAIN_GPUS:-0 1}"
EVAL_GPUS="${EVAL_GPUS:-0,1}"
ONLY="${ONLY:-}"

say() { echo "[$(date +%F_%T)] $*" | tee -a "${PIPELOG}"; }
set_status() {
  echo "PHASE=$1 STATUS=$2 TS=$(date +%F_%T) NOTE=${3:-}" > "${STATUS_FILE}"
  say "PHASE=$1 STATUS=$2 ${3:-}"
}
fail() { set_status "$1" FAILED "${2:-}"; exit 1; }
want() { [[ -z "${ONLY}" || "${ONLY}" == "$1" ]]; }

say "pipeline start: train_gpus='${TRAIN_GPUS}' eval_gpus='${EVAL_GPUS}'"

if want train; then
  set_status train RUNNING
  TRAIN_GPUS="${TRAIN_GPUS}" bash "${SCRIPT_DIR}/run_training.sh" \
    >>"${PIPELOG}" 2>&1 || fail train "one or more training jobs failed"
  set_status train OK
fi

if want resolve; then
  set_status resolve RUNNING
  "${PY}" "${SCRIPT_DIR}/resolve_models.py" \
    --new-state-dir "${REPO_ROOT}/state" \
    >>"${PIPELOG}" 2>&1 || fail resolve "could not resolve all eight 40k checkpoints"
  count="$(grep -cve '^[[:space:]]*$' "${SCRIPT_DIR}/model_list.txt")"
  [[ "${count}" == "8" ]] || fail resolve "model list has ${count} rows, expected 8"
  set_status resolve OK "8 checkpoints"
fi

if want evaluate; then
  set_status evaluate RUNNING
  MODEL_LIST="${SCRIPT_DIR}/model_list.txt" \
  EVAL_GPUS="${EVAL_GPUS}" \
    bash "${SCRIPT_DIR}/run_eval_sweep.sh" \
    >>"${PIPELOG}" 2>&1 || fail evaluate "classification/regression sweep failed"
  grep -q NM_SINGLE_SOURCE_DOWNSTREAM_EVAL_DONE "${PIPELOG}" \
    || fail evaluate "eval completion marker absent"
  set_status evaluate OK
fi

if want assemble; then
  set_status assemble RUNNING
  "${PY}" "${ANALYSIS_DIR}/assemble_results.py" \
    >>"${PIPELOG}" 2>&1 || fail assemble "missing or invalid downstream cells"
  set_status assemble OK
fi

if want plot; then
  set_status plot RUNNING
  "${PY}" "${ANALYSIS_DIR}/plot_results.py" \
    >>"${PIPELOG}" 2>&1 || fail plot "figure generation failed"
  set_status plot OK
fi

set_status pipeline OK
