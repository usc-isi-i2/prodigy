#!/usr/bin/env bash
# Evaluate each fill-in rung (4src..7src) on ALL 8 test graphs (NM, 30-way, 3-shot)
# -> the eval log dirs that assemble_full_table.py reads. Requires model_list.txt
# (see make_model_list.sh). Same harness/flags as the ladder + single-source matrix,
# so the 4 new rows are directly comparable to the existing rungs 1/2/3/8.
#
#   ./eval_ladder_tucker.sh                       # 32 NM eval jobs (4 rungs x 8 graphs), serial
#   GPUS="0,1,2,3" ./eval_ladder_tucker.sh        # fan out across 4 GPUs
#   DATA_ROOT=/some/other/data ./eval_ladder_tucker.sh     # override graph root
#   ./eval_ladder_tucker.sh --dry-run             # preview jobs
#
# DATA_ROOT defaults to the canonical /dataMeR1/phil/data catalog root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"

python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
  --model-list "${SCRIPT_DIR}/model_list.txt" \
  --data-root "${DATA_ROOT:-/dataMeR1/phil/data}" \
  --datasets ukr_rus_twitter,covid19_twitter,midterm,covid_political,election2020,ukr_rus_suspended,twibot20,cp_hk_twitter \
  --tasks nm \
  --shots 3 \
  --nm-n-way 30 \
  --gpus "${GPUS:-}" \
  "$@"
