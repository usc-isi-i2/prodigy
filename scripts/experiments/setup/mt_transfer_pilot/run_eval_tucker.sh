#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
cd "${REPO_ROOT}"
python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
  --model-list "${SCRIPT_DIR}/model_list.txt" --python python3 \
  --data-root /dataMeR1/phil/data \
  --datasets covid_political,election2020,facebook_page_reference,twibot20,ukr_rus_suspended \
  --graph-filenames facebook_page_reference=page_reference_graph.pt \
  --tasks pl --shots 3 --pl-dataset-len-cap 25 --batch-size 4 --workers 2 \
  --gpus 2,3,2,3,2,3 --continue-on-error
