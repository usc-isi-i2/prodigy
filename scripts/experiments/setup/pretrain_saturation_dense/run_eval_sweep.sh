#!/usr/bin/env bash
# Score the 6 dense saturation checkpoints (100 and 500 per arm) on the same benchmark,
# with the same flags, as the twelve already-trained ones.
#
# This is a thin wrapper over ../pretrain_saturation_existing/run_eval_sweep.sh rather
# than a second copy of it: the two halves of the curve MUST be measured identically
# (same shots, same targets, same transform, same dataset sets), and two copies of a
# flag list drift. Only the model list differs.
#
#   bash run_eval_sweep.sh --gpus 0,1
#   DRY_RUN=1 bash run_eval_sweep.sh --gpus 0,1
#   WITH_NM=1 bash run_eval_sweep.sh --gpus 0,1
#
# 6 ckpts x (4 graphs x 2 targets + 4 graphs) = 72 jobs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED="${SCRIPT_DIR}/../pretrain_saturation_existing/run_eval_sweep.sh"
[[ -f "${SHARED}" ]] || { echo "shared sweep script not found: ${SHARED}" >&2; exit 2; }

ML="${MODEL_LIST:-${SCRIPT_DIR}/model_list.txt}"
[[ -f "${ML}" ]] || { echo "model list not found: ${ML} (run make_model_list.py)" >&2; exit 2; }

MODEL_LIST="${ML}" exec bash "${SHARED}" "$@"
