#!/usr/bin/env bash
# Score the 12 already-trained saturation checkpoints on the downstream benchmark.
#
#   * node regression      -- 10-shot, log1p, followers_count + account_age_days
#                             on the 4 graphs carrying profile targets
#   * node classification  -- 10-shot, on the 4 labeled graphs
#
# 12 ckpts x (4 graphs x 2 targets + 4 graphs) = 144 jobs. The dense half adds 72 more
# for a 216-job, 18-checkpoint curve.
#
# Two targets, not the three used by nm_ladder_downstream: followers_count is the
# topology-explainable one (tracks in-degree), account_age_days has no topological route
# at all, so the pair brackets the range. statuses_count is another scale measure and
# would retrace the followers curve. Both are in the existing sweeps' target set, so
# these rows stay comparable to the ladder arms in the shared CSVs.
#
# Neighbor matching is OFF by default -- this experiment asks when DOWNSTREAM transfer
# saturates, and NM is the pretraining objective itself. Set WITH_NM=1 to add it
# (+8 jobs/ckpt), which makes the rows directly comparable to the NM ladder tables.
#
# Static LP does NOT go through this runner: its episodic slp path is void (center-blind
# scoring, frozen random prototypes, degree-confounded negatives -- see AGENTS.md and
# analysis/objectives/multitask_ssl/multitask_ssl/FINDINGS_rescore.md). Temporal LP has the same unrepaired defect.
# Both are deliberately out of scope.
#
# Usage (Tucker, from the worktree holding this branch):
#   bash run_eval_sweep.sh --gpus 0,1
#   DRY_RUN=1 bash run_eval_sweep.sh --gpus 0,1     # print the runner invocations only
#   WITH_NM=1 bash run_eval_sweep.sh --gpus 0,1
#
# GPUs: checked 2026-07-27 -- a vLLM worker is pinned across 2 AND 3 (~76 GB each), so
# only 0 and 1 are ours. Verify before launching and override:
#   ssh tucker nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
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
NM_DATASETS="${NM_DATASETS:-ukr_rus_twitter,covid19_twitter,midterm,covid_political,election2020,ukr_rus_suspended,twibot20,cp_hk_twitter}"
REG_TARGETS="${REG_TARGETS:-followers_count,account_age_days}"

run() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY_RUN: '; printf '%q ' "$@"; printf '\n'
  else
    "$@"
  fi
}

n_models=$(grep -cvE '^\s*(#|$)' "${ML}" || true)
echo "=== node regression (${n_models} ckpts x 4 graphs x 2 targets) ==="
run python3 "${RUNNER}" "${COMMON[@]}" \
  --datasets "${REG_DATASETS}" \
  --tasks reg --shots 10 --reg-transform log1p --reg-targets "${REG_TARGETS}" "$@"

echo "=== node classification (${n_models} ckpts x 4 graphs) ==="
run python3 "${RUNNER}" "${COMMON[@]}" \
  --datasets "${PL_DATASETS}" \
  --tasks pl --shots 10 "$@"

if [[ "${WITH_NM:-0}" == "1" ]]; then
  echo "=== neighbor matching (${n_models} ckpts x 8 graphs, 30-way 3-shot) ==="
  run python3 "${RUNNER}" "${COMMON[@]}" \
    --datasets "${NM_DATASETS}" \
    --tasks nm --shots 3 --nm-n-way 30 "$@"
fi

# Parse reg + pl into the shared plotting CSVs, keyed by model = our sat_<arm>_s<step> key.
# --log-root MUST derive from THIS tree's REPO_ROOT: eval writes to <cwd>/log and we cd'd
# to REPO_ROOT above, so hardcoding the main checkout silently parses the wrong logs when
# the sweep runs from an isolated worktree.
#
# NEVER add --overwrite here. This sweep produces regression and classification logs and
# no static-LP ones, and this worktree's log/ holds only our own runs. Under the parser's
# pre-490af96 behaviour (rebuild each CSV from --log-root alone, write unconditionally)
# that combination truncated static_link_prediction.csv from 149 rows to zero and dropped
# every historical arm from the other two. The parser now merges by default and skips a
# task whose CSV has no matching run dir; --overwrite restores the destructive rebuild and
# is only correct when --log-root really holds every arm ever run.
echo "=== parse into shared CSVs ==="
run python3 scripts/harness/benchmark_tasks/parse_benchmark_eval_logs.py \
  --log-root "${REPO_ROOT}/log" --out-dir scripts/experiments/analysis/evaluation/shared_task_tables

# Neutral sentinel: pretrain_saturation_dense/run_eval_sweep.sh execs this same script
# with a different model list, so the marker must not claim which half just finished.
echo "PRETRAIN_SATURATION_EVAL_SWEEP_DONE (model_list=${ML})"
