#!/usr/bin/env bash
# Build an isolated Ukraine-suspended target-enriched graph and run three floors.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ANALYSIS_DIR="${REPO_ROOT}/scripts/experiments/analysis/nm_single_source_downstream"
CANONICAL_DATA_ROOT="${CANONICAL_DATA_ROOT:-/dataMeR1/phil/data}"
BASELINE_DATA_ROOT="${BASELINE_DATA_ROOT:-${REPO_ROOT}/state/nmssd_baseline_data}"
TARGETS="followers_count,statuses_count,account_age_days"
DATASETS="ukr_rus_suspended,twibot20"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
PY="${PY:-python3}"

mkdir -p "${BASELINE_DATA_ROOT}/ukr_rus_suspended/graphs"
mkdir -p "${BASELINE_DATA_ROOT}/twibot20"
ln -sfn "${CANONICAL_DATA_ROOT}/twibot20/graphs" \
  "${BASELINE_DATA_ROOT}/twibot20/graphs"

"${PY}" "${SCRIPT_DIR}/enrich_ukr_suspended_targets.py" \
  --graph "${CANONICAL_DATA_ROOT}/ukr_rus_suspended/graphs/retweet_graph.pt" \
  --user-csv "${CANONICAL_DATA_ROOT}/social_llm_data/ukr_rus_suspended/user_data.csv" \
  --out "${BASELINE_DATA_ROOT}/ukr_rus_suspended/graphs/retweet_graph.pt" \
  --overwrite

"${PY}" "${REPO_ROOT}/scripts/experiments/setup/topology_feature_ssl/leakage_baseline.py" \
  --features raw --data-root "${BASELINE_DATA_ROOT}" \
  --datasets "${DATASETS}" --targets "${TARGETS}" \
  --shots 10 --n-query 12 --episodes 500 --transform log1p --skip-fulldata \
  --out "${ANALYSIS_DIR}/data/regression_baseline_raw_features.csv"

"${PY}" "${REPO_ROOT}/scripts/experiments/setup/topology_feature_ssl/leakage_baseline.py" \
  --features structural --mode directed3 --data-root "${BASELINE_DATA_ROOT}" \
  --datasets "${DATASETS}" --targets "${TARGETS}" \
  --shots 10 --n-query 12 --episodes 500 --transform log1p --skip-fulldata \
  --out "${ANALYSIS_DIR}/data/regression_baseline_raw_degree.csv"

"${PY}" "${REPO_ROOT}/scripts/eval/eval_ckpts_all_graph_tasks_tucker.py" \
  --model-list "${SCRIPT_DIR}/random_init_model_list.txt" \
  --python "${PY}" --data-root "${BASELINE_DATA_ROOT}" \
  --datasets "${DATASETS}" --tasks regression --shots 10 \
  --reg-targets "${TARGETS}" --reg-transform log1p \
  --gpus "${EVAL_GPUS:-0,1}" --continue-on-error

"${PY}" "${REPO_ROOT}/scripts/harness/benchmark_tasks/parse_benchmark_eval_logs.py" \
  --log-root "${REPO_ROOT}/log" \
  --out-dir "${ANALYSIS_DIR}/data/regression_baseline_random_init_parsed" \
  --reg-glob "eval_random_init_to_*_reg_*" \
  --slp-glob "__no_static_lp_runs__" \
  --pl-glob "__no_classification_runs__" \
  --overwrite

"${PY}" "${ANALYSIS_DIR}/assemble_regression_baselines.py"
"${PY}" "${ANALYSIS_DIR}/plot_regression_baselines.py"

echo "NMSSD_REGRESSION_BASELINES_DONE"
