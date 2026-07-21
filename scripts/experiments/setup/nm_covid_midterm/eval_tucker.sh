#!/usr/bin/env bash
# Evaluate all 5 regimes on covid + midterm + merged (NM, 3-shot, 30-way).
# Requires model_list.txt (see make_model_list.sh).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"

python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
  --model-list "${SCRIPT_DIR}/model_list.txt" \
  --datasets covid19_twitter,midterm,ukr_rus_twitter,merged_covid_midterm \
  --tasks nm \
  --shots 3 \
  --nm-n-way 30 \
  "$@"
