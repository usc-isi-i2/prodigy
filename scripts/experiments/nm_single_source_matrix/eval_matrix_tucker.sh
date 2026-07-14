#!/usr/bin/env bash
# Evaluate every trained single-source model on all 8 test graphs (NM, 30-way,
# 3-shot) -> the 8x8 of eval log dirs that build_ss_matrix.py reads.
# Requires model_list.txt (see make_model_list.sh).
#
#   ./eval_matrix_tucker.sh                       # 64 NM eval jobs, serial
#   GPUS="0,1,2,3" ./eval_matrix_tucker.sh        # fan out across 4 GPUs
#   DATA_ROOT=/dataMeR2/phil/data ./eval_matrix_tucker.sh   # override graph root
#   ./eval_matrix_tucker.sh --dry-run             # preview jobs
#
# NOTE on DATA_ROOT: the training configs and the NM ladder use /dataMeR1/phil/data.
# The shared eval harness defaults to /dataMeR2/phil/data. We pass /dataMeR1 here to
# match the ladder; override with DATA_ROOT= if your graphs live elsewhere.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"

python3 scripts/experiments/eval/eval_ckpts_all_graph_tasks_tucker.py \
  --model-list "${SCRIPT_DIR}/model_list.txt" \
  --data-root "${DATA_ROOT:-/dataMeR1/phil/data}" \
  --datasets ukr_rus_twitter,covid19_twitter,midterm,covid_political,election2020,ukr_rus_suspended,twibot20,cp_hk_twitter \
  --tasks nm \
  --shots 3 \
  --nm-n-way 30 \
  --gpus "${GPUS:-}" \
  "$@"
