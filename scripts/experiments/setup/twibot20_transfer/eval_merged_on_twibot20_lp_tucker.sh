#!/usr/bin/env bash
# Experiment (b), LINEAR-PROBE variant: evaluate the merged-vs-single NM study
# checkpoints on TwiBot-20 bot-vs-human classification as a linear probe
# (--linear_probe True, all classes per episode, many-shot support) instead of
# the near-chance 3-shot readout. This measures representation quality.
#
# Requires model_list_merged.txt (run make_model_list_merged.sh first).
#   SHOTS=20 bash eval_merged_on_twibot20_lp_tucker.sh --gpus 0,1,2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SHOTS="${SHOTS:-20}"           # support examples per class for the probe
TRAIN_CAP="${TRAIN_CAP:-0}"    # 0 = uncapped labeled pool

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"

python3 scripts/experiments/eval/eval_ckpts_all_graph_tasks_tucker.py \
  --model-list "${SCRIPT_DIR}/model_list_merged.txt" \
  --datasets twibot20 \
  --tasks classification \
  --shots "${SHOTS}" \
  --pl-linear-probe \
  --pl-train-cap "${TRAIN_CAP}" \
  "$@"
