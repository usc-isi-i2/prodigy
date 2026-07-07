#!/usr/bin/env bash
# Feature-ablation sweep: for a fixed checkpoint, evaluate each task with node
# features INTACT vs. ZEROED vs. PERMUTED. The intact-minus-ablated accuracy gap
# measures how much each task (NM, LP, classification) relies on node features
# vs. graph topology. See README.md.
#
# Usage (Tucker):
#   MODEL_LIST=path/to/model_list.txt bash run_feature_ablation_tucker.sh --gpus 0,1
# Any extra args (--datasets, --tasks, --shots, --nm-n-way, --gpus, ...) are
# forwarded to eval_ckpts_all_graph_tasks_tucker.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODEL_LIST="${MODEL_LIST:-${SCRIPT_DIR}/model_list.txt}"
DATASETS="${DATASETS:-midterm,covid19_twitter,ukr_rus_twitter,twibot20}"
TASKS="${TASKS:-neighbor_matching,temporal_link_prediction,classification}"
SHOTS="${SHOTS:-3}"
MODES="${MODES:-none zero permute}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"

if [[ ! -f "${MODEL_LIST}" ]]; then
  echo "MODEL_LIST not found: ${MODEL_LIST}" >&2
  echo "Create it as '<model_name> <checkpoint_path.pt>' lines (see README.md)." >&2
  exit 1
fi

for mode in ${MODES}; do
  echo "===================================================================="
  echo "[feature-ablation] mode=${mode}  datasets=${DATASETS}  tasks=${TASKS}  shots=${SHOTS}"
  echo "===================================================================="
  python3 scripts/experiments/eval/eval_ckpts_all_graph_tasks_tucker.py \
    --model-list "${MODEL_LIST}" \
    --datasets "${DATASETS}" \
    --tasks "${TASKS}" \
    --shots "${SHOTS}" \
    --ablate-features "${mode}" \
    "$@"
done

echo "[feature-ablation] done. Parse with:"
echo "  python3 ${SCRIPT_DIR}/parse_feature_ablation.py --log-root log --out ${SCRIPT_DIR}/feature_ablation_results.csv"
