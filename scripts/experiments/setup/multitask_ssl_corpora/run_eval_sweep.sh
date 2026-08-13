#!/usr/bin/env bash
# Frozen-encoder benchmark sweep for multitask_ssl_corpora: freeze each of the 8
# encoders (2 corpora x {NM,CL,FP,MIX}) and run the joint benchmark, identical in
# tasks/datasets to the multitask_ssl_rotation sweep so the headline contrast
# (MIX above-chance static-LP, controls at chance) is directly comparable per corpus:
#   * node regression   — 10-shot, log1p, followers/statuses/account_age
#   * static link pred  — zero-shot, --slp-n-query 4
#   * node classification (pl) — 10-shot (auto-gated to labeled graphs)
# over the focused 5 datasets: midterm, ukr_rus_twitter, covid19_twitter,
# twibot20, election2020. All 8 arms are bio-768/mean-SAGE — no structural args.
# Results land in the shared plotting CSVs keyed by model = <corpus>_<ARM>.
#
# Usage (Tucker, from the worktree holding this branch, ideally via the watcher):
#   MODEL_LIST=scripts/experiments/setup/multitask_ssl_corpora/model_list.txt \
#     bash scripts/experiments/setup/multitask_ssl_corpora/run_eval_sweep.sh --gpus 0,1,2,3
# Requires conda's bin on PATH (export PATH="/home/mhchu/miniconda3/bin:$PATH")
# when run from a detached/non-interactive shell.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"   # setup/<name> is 4 levels below repo root
ML="${MODEL_LIST:-${SCRIPT_DIR}/model_list.txt}"
[[ -f "${ML}" ]] || { echo "model list not found: ${ML} (run make_model_list.sh)" >&2; exit 2; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"

RUNNER=scripts/eval/eval_ckpts_all_graph_tasks_tucker.py
COMMON=(--model-list "${ML}" --python python3
        --data-root /dataMeR1/phil/data
        --datasets midterm,ukr_rus_twitter,covid19_twitter,twibot20,election2020
        --continue-on-error)

# node regression (feature-channel read)
python3 "${RUNNER}" "${COMMON[@]}" --tasks reg --shots 10 --reg-transform log1p \
  --reg-targets followers_count,statuses_count,account_age_days "$@"

# static link prediction (the emergent-capability headline; zero-shot)
python3 "${RUNNER}" "${COMMON[@]}" --tasks slp --shots 0 --slp-n-query 4 "$@"

# node classification (feature task; auto-gated to labeled graphs)
python3 "${RUNNER}" "${COMMON[@]}" --tasks pl --shots 10 "$@"

# parse reg + slp + pl into the shared plotting CSVs (keyed by model = corpus_arm).
# --log-root MUST derive from THIS tree's REPO_ROOT (eval writes to <cwd>/log and we
# cd'd to REPO_ROOT above); hardcoding the main tree silently parses the wrong logs
# when the sweep runs from an isolated worktree (e.g. /dataMeR1/phil/gfm/prodigy-msc).
python3 scripts/harness/benchmark_tasks/parse_benchmark_eval_logs.py \
  --log-root "${REPO_ROOT}/log" --out-dir scripts/experiments/analysis/evaluation/shared_task_tables

echo "MULTITASK_SSL_CORPORA_EVAL_SWEEP_DONE"
