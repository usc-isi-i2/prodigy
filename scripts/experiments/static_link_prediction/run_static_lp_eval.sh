#!/usr/bin/env bash
# Evaluate checkpoints on the static link-prediction benchmark task. Thin wrapper
# over the shared eval runner with --tasks slp.
#
# Prerequisite: graphs must carry static_background/static_holdout edge views (run
# enrich_all_graphs.sh once, or rebuild graphs — views are emitted by default).
#
# Pass a checkpoint source and any overrides through "$@", e.g.:
#   bash run_static_lp_eval.sh \
#     --checkpoint-run-dir /dataMeR2/phil/gfm/prodigy/state/<run> \
#     --datasets midterm,ukr_rus_twitter,covid19_twitter,cp_hk_twitter,twibot20 \
#     --gpus 0,1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"

# Retweet graphs are sparse: a static-LP episode needs (n_shots + n_query) held-out
# edges on one center, so use zero-shot + small n_query so low-degree centers qualify.
# (twibot20 is dense enough for a few-shot variant; add --shots 0,3 if desired.)
python3 scripts/experiments/eval/eval_ckpts_all_graph_tasks_tucker.py \
  --tasks slp \
  --datasets midterm,ukr_rus_twitter,covid19_twitter,cp_hk_twitter,twibot20 \
  --slp-hard-negatives True \
  --slp-n-query 4 \
  --shots 0 \
  "$@"
