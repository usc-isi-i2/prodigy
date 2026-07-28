#!/usr/bin/env bash
# Downstream benchmark sweep for the NM ladder: freeze each of the 21 distinct rung
# encoders and run the frozen-encoder benchmark on every task the graphs support.
#
#   * node regression      -- 10-shot, log1p, followers/statuses/account_age
#                             on the 4 graphs carrying profile targets
#   * node classification  -- 10-shot, on the 4 labeled graphs
#
# Static link prediction is NOT run here. The runner's episodic slp path is void
# (center-blind scoring, frozen random prototypes, degree-confounded negatives -- see
# AGENTS.md and analysis/multitask_ssl/FINDINGS_rescore.md); sLP goes through
# run_pair_lp_sweep.sh instead. Temporal LP has the same unrepaired defect and is
# deliberately out of scope.
#
# Shots/targets/transform match scripts/experiments/setup/multitask_ssl_corpora/
# run_eval_sweep.sh exactly, so the ladder rows land in the same shared CSVs directly
# comparable to the existing cov_*/all8_*/B0/B1/E1 arms.
#
# Usage (Tucker, from the worktree holding this branch, ideally via run_pipeline_tucker.sh):
#   MODEL_LIST=.../model_list.txt bash run_eval_sweep.sh --gpus 0,1,2
#   DRY_RUN=1 bash run_eval_sweep.sh --gpus 0,1,2     # print the runner invocations only
#
# Requires conda's bin on PATH (export PATH="/home/mhchu/miniconda3/bin:$PATH") when run
# from a detached/non-interactive shell -- see AGENTS.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"   # setup/<name> is 4 levels below repo root
ML="${MODEL_LIST:-${SCRIPT_DIR}/model_list.txt}"
[[ -f "${ML}" ]] || { echo "model list not found: ${ML} (run make_model_list.py)" >&2; exit 2; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"

RUNNER=scripts/eval/eval_ckpts_all_graph_tasks_tucker.py
COMMON=(--model-list "${ML}" --python python3
        --data-root "${DATA_ROOT:-/dataMeR1/phil/data}"
        --continue-on-error)

# Explicit per-task dataset sets. The runner auto-gates unsupported tasks, but gating
# means LOADING the artifact first -- passing all 8 would page in the 23M-node covid
# graph only to skip it.
REG_DATASETS="${REG_DATASETS:-ukr_rus_twitter,covid19_twitter,midterm,twibot20}"
PL_DATASETS="${PL_DATASETS:-covid_political,election2020,ukr_rus_suspended,twibot20}"
REG_TARGETS="${REG_TARGETS:-followers_count,statuses_count,account_age_days}"

run() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY_RUN: '; printf '%q ' "$@"; printf '\n'
  else
    "$@"
  fi
}

echo "=== node regression (21 models x 4 graphs x 3 targets = 252 jobs) ==="
run python3 "${RUNNER}" "${COMMON[@]}" \
  --datasets "${REG_DATASETS}" \
  --tasks reg --shots 10 --reg-transform log1p --reg-targets "${REG_TARGETS}" "$@"

echo "=== node classification (21 models x 4 graphs = 84 jobs) ==="
run python3 "${RUNNER}" "${COMMON[@]}" \
  --datasets "${PL_DATASETS}" \
  --tasks pl --shots 10 "$@"

# Parse reg + pl into the shared plotting CSVs, keyed by model = the run prefix.
# --log-root MUST derive from THIS tree's REPO_ROOT: eval writes to <cwd>/log and we
# cd'd to REPO_ROOT above, so hardcoding the main checkout silently parses the wrong
# logs when the sweep runs from an isolated worktree.
echo "=== parse into shared CSVs ==="
run python3 scripts/harness/benchmark_tasks/parse_benchmark_eval_logs.py \
  --log-root "${REPO_ROOT}/log" --out-dir scripts/experiments/analysis

echo "NM_LADDER_DOWNSTREAM_EVAL_SWEEP_DONE"
