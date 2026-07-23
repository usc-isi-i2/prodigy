#!/usr/bin/env bash
# Evaluate each trained rung on ALL 8 test graphs (NM, 30-way, 3-shot). Same harness and
# flags as the published ladder + single-source matrix, so the new rows are directly
# comparable to the existing rungs.
#
#   ./eval_ladder_tucker.sh                    # 88 NM eval jobs (11 rungs x 8 graphs)
#   GPUS="0,1,2,3" ./eval_ladder_tucker.sh     # fan out across 4 GPUs
#   GATE=1 GPUS="0,1,2,3" ./eval_ladder_tucker.sh   # 8 jobs for the gate run only
#   DATA_ROOT=/some/other/data ./eval_ladder_tucker.sh
#   ./eval_ladder_tucker.sh --dry-run          # preview jobs
#
# Requires model_list.txt (or model_list_gate.txt) from make_model_list.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

if [[ "${GATE:-0}" == "1" ]]; then
  MODEL_LIST="${SCRIPT_DIR}/model_list_gate.txt"
else
  MODEL_LIST="${SCRIPT_DIR}/model_list.txt"
fi
[[ -f "${MODEL_LIST}" ]] || { echo "missing ${MODEL_LIST} -- run make_model_list.sh first" >&2; exit 2; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"

python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
  --model-list "${MODEL_LIST}" \
  --data-root "${DATA_ROOT:-/dataMeR1/phil/data}" \
  --datasets ukr_rus_twitter,covid19_twitter,midterm,covid_political,election2020,ukr_rus_suspended,twibot20,cp_hk_twitter \
  --tasks nm \
  --shots 3 \
  --nm-n-way 30 \
  --gpus "${GPUS:-}" \
  "$@"
