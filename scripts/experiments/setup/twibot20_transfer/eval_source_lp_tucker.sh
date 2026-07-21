#!/usr/bin/env bash
# Experiment (a), LINEAR-PROBE variant: evaluate the TwiBot-20-trained NM
# checkpoint on every labeled target as a classification linear probe
# (--linear_probe True, many-shot support). Unlabeled graphs are auto-skipped.
#
# Requires model_list_source.txt (run make_model_list_source.sh first).
#   SHOTS=20 bash eval_source_lp_tucker.sh --gpus 0,1,2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SHOTS="${SHOTS:-20}"
TRAIN_CAP="${TRAIN_CAP:-0}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"

python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
  --model-list "${SCRIPT_DIR}/model_list_source.txt" \
  --datasets twibot20,covid_political,election2020,ukr_rus_suspended,cp_hk_twitter \
  --tasks classification \
  --shots "${SHOTS}" \
  --pl-linear-probe \
  --pl-train-cap "${TRAIN_CAP}" \
  "$@"
