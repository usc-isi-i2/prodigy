#!/usr/bin/env bash
# Free preview for the topology_feature_ssl experiment (see README "Budget").
#
# Reads a prior we already own: eval the EXISTING covid task_transfer nm & fp
# checkpoints on NODE REGRESSION and compare Spearman. fp (masked feature
# prediction) approximates E3's objective, so if fp already beats nm on
# regression, E3's core hypothesis is pre-validated before any new pretrain.
#
# No new training. Frozen-encoder eval only. Run on Tucker (prodigy env, GPUs).
# Pass overrides through "$@" (e.g. --gpus 0,1).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"

RUNNER=scripts/experiments/eval/eval_ckpts_all_graph_tasks_tucker.py
ML="${SCRIPT_DIR}/model_list_free_preview.txt"

# Focused 5 datasets: 3 in-domain (ukr_rus, covid, midterm) + 2 held-out
# (twibot20, election2020). 3 representative regression targets: followers
# (influence), statuses (activity), account_age_days (age).
python3 "${RUNNER}" \
  --model-list "${ML}" --python python3 \
  --data-root /dataMeR1/phil/data \
  --datasets midterm,ukr_rus_twitter,covid19_twitter,twibot20,election2020 \
  --tasks reg --shots 10 --reg-transform log1p \
  --reg-targets followers_count,statuses_count,account_age_days \
  --continue-on-error "$@"

# Parse reg logs into the shared plotting CSV (keyed by model = strategy).
python3 scripts/analysis/benchmark_tasks/parse_benchmark_eval_logs.py \
  --log-root /dataMeR1/phil/gfm/prodigy/log --out-dir scripts/plotting

# Print the nm-vs-fp Spearman comparison table.
python3 "${SCRIPT_DIR}/compare_free_preview.py" \
  --csv scripts/plotting/node_regression/data/node_regression.csv

echo "TOPOLOGY_FEATURE_SSL_FREE_PREVIEW_DONE"
