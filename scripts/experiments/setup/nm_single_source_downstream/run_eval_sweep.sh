#!/usr/bin/env bash
# Evaluate all eight single-source NM encoders on valid classification/regression tasks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ANALYSIS_DIR="${REPO_ROOT}/scripts/experiments/analysis/transfer/matrices/prodigy_nm/downstream/nm_single_source_downstream"
MODEL_LIST="${MODEL_LIST:-${SCRIPT_DIR}/model_list.txt}"
[[ -f "${MODEL_LIST}" ]] || { echo "missing model list: ${MODEL_LIST}" >&2; exit 2; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"

RUNNER="scripts/eval/eval_ckpts_all_graph_tasks_tucker.py"
COMMON=(
  --model-list "${MODEL_LIST}"
  --python python3
  --data-root "${DATA_ROOT:-/dataMeR1/phil/data}"
  --continue-on-error
)
REG_DATASETS="${REG_DATASETS:-ukr_rus_twitter,covid19_twitter,midterm,twibot20}"
CLASS_DATASETS="${CLASS_DATASETS:-covid_political,election2020,ukr_rus_suspended,twibot20}"
REG_TARGETS="${REG_TARGETS:-followers_count,statuses_count,account_age_days}"

run() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY_RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run python3 "${RUNNER}" "${COMMON[@]}" \
  --datasets "${REG_DATASETS}" \
  --tasks regression --shots 10 --reg-transform log1p \
  --reg-targets "${REG_TARGETS}" --gpus "${EVAL_GPUS:-0,1}" "$@"

run python3 "${RUNNER}" "${COMMON[@]}" \
  --datasets "${CLASS_DATASETS}" \
  --tasks classification --shots 10 --gpus "${EVAL_GPUS:-0,1}" "$@"

# Parse only this worktree's logs into experiment-owned raw tables. --overwrite is
# safe here because the output is not one of the shared append-only task CSVs.
run python3 scripts/harness/benchmark_tasks/parse_benchmark_eval_logs.py \
  --log-root "${REPO_ROOT}/log" \
  --out-dir "${ANALYSIS_DIR}/data/parsed" \
  --overwrite

echo "NM_SINGLE_SOURCE_DOWNSTREAM_EVAL_DONE"
