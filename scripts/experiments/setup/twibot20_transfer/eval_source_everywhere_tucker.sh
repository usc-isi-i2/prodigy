#!/usr/bin/env bash
# Experiment (a): evaluate the TwiBot-20-trained NM checkpoint on every other
# graph and task (NM + classification; LP is auto-skipped where unsupported).
# Requires model_list_source.txt (run make_model_list_source.sh first).
# Pass --gpus 0,1,2 to parallelize.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"

# All graphs except twibot20 itself (add twibot20 to --datasets for a self-eval).
python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
  --model-list "${SCRIPT_DIR}/model_list_source.txt" \
  --datasets midterm,covid19_twitter,ukr_rus_twitter,merged_ukr_rus_covid,merged_covid_midterm,covid_political,election2020,ukr_rus_suspended,twibot20 \
  --tasks nm,classification \
  --shots 3 \
  --nm-n-way 30 \
  "$@"
