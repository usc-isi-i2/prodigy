#!/usr/bin/env bash
# Per-encoder benchmark sweep for multitask_ssl_rotation. Freeze each arm's encoder
# (NM/CL/FP/MIX) and run the joint benchmark across ALL graphs and ALL downstream
# tasks, so the comparison is a table subtraction MIX - max(NM, CL, FP):
#   * node regression   — 10-shot, log1p, 3 representative targets
#                         (followers=influence, statuses=activity, account_age=age)
#   * static link pred   — zero-shot + --slp-n-query 4 (sparse-graph safe)
#   * node classification (pl) — 10-shot (auto-gated to graphs with labels)
# over the focused 5 datasets: 3 in-domain (ukr_rus, covid, midterm) + 2 held-out
# (twibot20, election2020). All four arms are bio-768/mean-SAGE, so NO structural
# args are needed. Reuses the shared eval driver + parser; results land in the
# shared plotting CSVs keyed by model = arm.
#
# Usage (Tucker, prodigy env, tmux):
#   MODEL_LIST=scripts/experiments/multitask_ssl_rotation/model_list.txt \
#     bash scripts/experiments/multitask_ssl_rotation/run_eval_sweep.sh --gpus 0,1,2,3
# Build MODEL_LIST first with make_model_list.sh. Pass overrides through "$@".
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ML="${MODEL_LIST:-${SCRIPT_DIR}/model_list.txt}"
[[ -f "${ML}" ]] || { echo "model list not found: ${ML} (run make_model_list.sh)" >&2; exit 2; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"

RUNNER=scripts/experiments/eval/eval_ckpts_all_graph_tasks_tucker.py
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

# parse reg + slp into the shared plotting CSVs (keyed by model = arm)
python3 scripts/analysis/benchmark_tasks/parse_benchmark_eval_logs.py \
  --log-root /dataMeR1/phil/gfm/prodigy/log --out-dir scripts/plotting

echo "MULTITASK_SSL_ROTATION_EVAL_SWEEP_DONE"
