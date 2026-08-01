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
SEEDS="${SEEDS:-0,1,2,3,4}"
DRY_RUN="${DRY_RUN:-0}"

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf "[dry-run]"
    printf " %q" "$@"
    printf "\n"
  else
    "$@"
  fi
}

if [[ "${DRY_RUN}" == "1" ]]; then
  PY="${PY:-python3}"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate prodigy
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  PY="${PY:-python3}"

  mkdir -p "${BASELINE_DATA_ROOT}/ukr_rus_suspended/graphs"
  mkdir -p "${BASELINE_DATA_ROOT}/twibot20"
  ln -sfn "${CANONICAL_DATA_ROOT}/twibot20/graphs" \
    "${BASELINE_DATA_ROOT}/twibot20/graphs"
fi

run_cmd "${PY}" "${SCRIPT_DIR}/enrich_ukr_suspended_targets.py" \
  --graph "${CANONICAL_DATA_ROOT}/ukr_rus_suspended/graphs/retweet_graph.pt" \
  --user-csv "${CANONICAL_DATA_ROOT}/social_llm_data/ukr_rus_suspended/user_data.csv" \
  --out "${BASELINE_DATA_ROOT}/ukr_rus_suspended/graphs/retweet_graph.pt" \
  --overwrite

IFS=',' read -r -a SEED_VALUES <<< "${SEEDS}"
for seed in "${SEED_VALUES[@]}"; do
  if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed '${seed}' in SEEDS=${SEEDS}" >&2
    exit 2
  fi
  episode_seed=$((448 + seed))  # sum(ord(c) for c in "test") + seed offset

  run_cmd "${PY}" "${REPO_ROOT}/scripts/experiments/setup/topology_feature_ssl/leakage_baseline.py" \
    --features raw --data-root "${BASELINE_DATA_ROOT}" \
    --datasets "${DATASETS}" --targets "${TARGETS}" \
    --shots 10 --n-query 12 --episodes 500 --transform log1p --skip-fulldata \
    --seed "${episode_seed}" \
    --out "${ANALYSIS_DIR}/data/regression_baseline_raw_features_seed${seed}.csv"

  run_cmd "${PY}" "${REPO_ROOT}/scripts/experiments/setup/topology_feature_ssl/leakage_baseline.py" \
    --features structural --mode directed3 --data-root "${BASELINE_DATA_ROOT}" \
    --datasets "${DATASETS}" --targets "${TARGETS}" \
    --shots 10 --n-query 12 --episodes 500 --transform log1p --skip-fulldata \
    --seed "${episode_seed}" \
    --out "${ANALYSIS_DIR}/data/regression_baseline_raw_degree_seed${seed}.csv"

  model_list="${BASELINE_DATA_ROOT}/random_init_seed${seed}_model_list.txt"
  if [[ "${DRY_RUN}" != "1" ]]; then
    printf "random_init_s%s NONE\n" "${seed}" > "${model_list}"
  fi
  run_cmd "${PY}" "${REPO_ROOT}/scripts/eval/eval_ckpts_all_graph_tasks_tucker.py" \
    --model-list "${model_list}" \
    --python "${PY}" --data-root "${BASELINE_DATA_ROOT}" \
    --datasets "${DATASETS}" --tasks regression --shots 10 \
    --reg-targets "${TARGETS}" --reg-n-query 12 --reg-transform log1p \
    --seed "${seed}" --eval-episode-seed-offset "${seed}" \
    --gpus "${EVAL_GPUS:-0,1}" --continue-on-error
done

run_cmd "${PY}" "${REPO_ROOT}/scripts/harness/benchmark_tasks/parse_benchmark_eval_logs.py" \
  --log-root "${REPO_ROOT}/log" \
  --out-dir "${ANALYSIS_DIR}/data/regression_baseline_random_init_parsed" \
  --reg-glob "eval_random_init_s*_to_*_reg_*" \
  --slp-glob "__no_static_lp_runs__" \
  --pl-glob "__no_classification_runs__" \
  --overwrite

run_cmd "${PY}" "${ANALYSIS_DIR}/assemble_regression_baselines.py" --seeds "${SEEDS}"
run_cmd "${PY}" "${ANALYSIS_DIR}/plot_regression_baselines.py"

echo "NMSSD_REGRESSION_BASELINES_DONE seeds=${SEEDS}"
