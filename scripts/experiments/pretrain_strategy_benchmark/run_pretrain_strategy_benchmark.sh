#!/usr/bin/env bash
# Compare pretraining strategies (different pretraining objectives on the same
# covid data) across the benchmark task families, using existing checkpoints.
#
# Prerequisite: graphs enriched with benchmark targets (node_targets + static
# views) — run scripts/graph_construction/enrich_all_graphs.sh once.
#
# Headline tasks are the diverse new ones (node regression + static LP); nm and
# classification are included for completeness. Results land in the shared
# per-task CSVs (keyed by model = strategy) and are compared in
# scripts/plotting/pretrain_strategy_benchmark/.
#
# Pass overrides through "$@" (e.g. --gpus 0,1).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"

RUNNER=scripts/experiments/eval/eval_ckpts_all_graph_tasks_tucker.py
ML="${SCRIPT_DIR}/model_list.txt"
COMMON=(--model-list "${ML}" --python python3
        --data-root /dataMeR1/phil/data
        --datasets midterm,ukr_rus_twitter,covid19_twitter,cp_hk_twitter,twibot20
        --continue-on-error)

# node regression (log1p, full profile panel; 10-shot)
python3 "${RUNNER}" "${COMMON[@]}" --tasks reg --shots 10 --reg-transform log1p "$@"

# static link prediction (zero-shot + small n_query so sparse graphs qualify)
python3 "${RUNNER}" "${COMMON[@]}" --tasks slp --shots 0 --slp-n-query 4 "$@"

# neighbor matching + classification baselines (3-shot)
python3 "${RUNNER}" "${COMMON[@]}" --tasks nm,pl --shots 3 "$@"

# parse the new-task results (reg + slp) into the plotting CSVs
python3 scripts/analysis/benchmark_tasks/parse_benchmark_eval_logs.py \
  --log-root /dataMeR1/phil/gfm/prodigy/log --out-dir scripts/plotting
echo "PRETRAIN_STRATEGY_BENCHMARK_DONE"
