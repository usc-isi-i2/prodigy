#!/usr/bin/env bash
# Experiment (b): evaluate the merged-vs-single NM study checkpoints on TwiBot-20
# (neighbor matching + bot-vs-human classification). Requires model_list_merged.txt
# (run make_model_list_merged.sh first). Pass --gpus 0,1,2 to parallelize.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"

python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
  --model-list "${SCRIPT_DIR}/model_list_merged.txt" \
  --datasets twibot20 \
  --tasks nm,classification \
  --shots 3 \
  --nm-n-way 30 \
  "$@"
