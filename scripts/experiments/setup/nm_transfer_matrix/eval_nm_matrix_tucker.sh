#!/usr/bin/env bash
# Evaluate all three trained models on all three test datasets (NM, zero-shot).
# Produces the 3x3 of eval log dirs that build_auc_matrix.py reads.
# Requires model_list.txt to be populated (see make_model_list.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"

python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
  --model-list "${SCRIPT_DIR}/model_list.txt" \
  --datasets ukr_rus_twitter,covid19_twitter,midterm \
  --tasks nm \
  --shots 3 \
  --nm-n-way 30 \
  "$@"
