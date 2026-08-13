#!/usr/bin/env bash
# Per-encoder benchmark sweep for multitask_ssl_pairs. Freeze each arm's encoder
# and run the joint benchmark across ALL graphs and ALL downstream tasks, so the
# comparison is a direct table read over the subset lattice of {NM, CL, FP}:
#   singles NM/CL/FP  ->  pairs NMCL/NMFP/CLFP  ->  triple MIX
# Tasks (identical to multitask_ssl_rotation, so pairs slot into the same table):
#   * node regression   — 10-shot, log1p, 3 representative targets
#                         (followers=influence, statuses=activity, account_age=age)
#   * static link pred   — zero-shot + --slp-n-query 4 (sparse-graph safe)
#   * node classification (pl) — 10-shot (auto-gated to graphs with labels)
# over the focused 5 datasets: 3 in-domain (ukr_rus, covid, midterm) + 2 held-out
# (twibot20, election2020). All arms are bio-768/mean-SAGE, so NO structural args
# are needed. Reuses the shared eval driver + parser; results land in the shared
# plotting CSVs keyed by model = arm.
#
# Build MODEL_LIST first with make_model_list.sh — point it at all 7 arms for the
# merged table (pairs + rotation controls) so every arm is scored identically here.
#
# Usage (Tucker, prodigy env, tmux):
#   MODEL_LIST=scripts/experiments/multitask_ssl_pairs/model_list.txt \
#     bash scripts/experiments/setup/multitask_ssl_pairs/run_eval_sweep.sh --gpus 0,1,2,3
# Pass overrides through "$@".
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
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

# node regression (headline for the feature-vs-topology read; NM is weak here)
python3 "${RUNNER}" "${COMMON[@]}" --tasks reg --shots 10 --reg-transform log1p \
  --reg-targets followers_count,statuses_count,account_age_days "$@"

# static link prediction (the direct topological task; zero-shot, small n_query)
python3 "${RUNNER}" "${COMMON[@]}" --tasks slp --shots 0 --slp-n-query 4 "$@"

# node classification (feature task; auto-gated to labeled graphs)
python3 "${RUNNER}" "${COMMON[@]}" --tasks pl --shots 10 "$@"

# parse reg + slp into the shared plotting CSVs (keyed by model = arm).
# --log-root MUST be this tree's own log/ (eval writes to <cwd>/log, and we cd'd to
# REPO_ROOT above). Hardcoding another tree silently parses the wrong logs when the
# sweep runs from an isolated worktree.
python3 scripts/harness/benchmark_tasks/parse_benchmark_eval_logs.py \
  --log-root "${REPO_ROOT}/log" --out-dir scripts/experiments/analysis/evaluation/shared_task_tables

echo "MULTITASK_SSL_PAIRS_EVAL_SWEEP_DONE"
