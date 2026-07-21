#!/usr/bin/env bash
# Evaluate checkpoints on the node-regression benchmark task (profile-attribute
# regression). Thin wrapper over the shared eval runner with --tasks reg.
#
# Prerequisite: graphs must carry node_targets (run enrich_all_graphs.sh once, or
# rebuild graphs with the generators — targets are emitted by default).
#
# Pass a checkpoint source and any overrides through "$@", e.g.:
#   bash run_node_regression_eval.sh \
#     --checkpoint-run-dir /dataMeR2/phil/gfm/prodigy/state/<run> \
#     --datasets midterm,ukr_rus_twitter,covid19_twitter,twibot20 \
#     --gpus 0,1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"

python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
  --tasks reg \
  --datasets midterm,ukr_rus_twitter,covid19_twitter,twibot20 \
  --reg-transform log1p \
  --shots 10 \
  "$@"
