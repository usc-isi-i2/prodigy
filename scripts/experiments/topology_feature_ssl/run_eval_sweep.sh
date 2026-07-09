#!/usr/bin/env bash
# Per-encoder benchmark sweep for topology_feature_ssl (see README "Eval").
# Freeze each arm's encoder and run the joint benchmark at a single shot setting:
#   * node regression  — 10-shot, log1p, 3 representative targets
#                        (followers=influence, statuses=activity, account_age=age)
#   * static link pred  — zero-shot + --slp-n-query 4 (sparse-graph safe)
#   * node classification (pl) — 10-shot (auto-gated to graphs with labels)
# over the focused 5 datasets: 3 in-domain (ukr_rus, covid, midterm) + 2 held-out
# (twibot20, election2020). Results land in the shared per-task plotting CSVs,
# keyed by model = arm (B0/B1/E1/...), so reading-chain deltas are table subtractions.
#
# Usage (Tucker, prodigy env, tmux):
#   MODEL_LIST=scripts/experiments/topology_feature_ssl/model_list.txt \
#     bash scripts/experiments/topology_feature_ssl/run_eval_sweep.sh --gpus 0,1,2,3
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

# node regression (headline for the objective axis; NM is weak here)
python3 "${RUNNER}" "${COMMON[@]}" --tasks reg --shots 10 --reg-transform log1p \
  --reg-targets followers_count,statuses_count,account_age_days "$@"

# static link prediction (the direct topological task; zero-shot, small n_query)
python3 "${RUNNER}" "${COMMON[@]}" --tasks slp --shots 0 --slp-n-query 4 "$@"

# node classification (feature task; auto-gated to labeled graphs)
python3 "${RUNNER}" "${COMMON[@]}" --tasks pl --shots 10 "$@"

# parse reg + slp into the shared plotting CSVs (keyed by model = arm)
python3 scripts/analysis/benchmark_tasks/parse_benchmark_eval_logs.py \
  --log-root /dataMeR1/phil/gfm/prodigy/log --out-dir scripts/plotting

echo "TOPOLOGY_FEATURE_SSL_EVAL_SWEEP_DONE"
