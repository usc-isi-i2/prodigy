#!/bin/bash
# Submit a model-list evaluation job.
#
# Usage: submit_eval.sh <dataset> <model_list.txt> [shots_csv]
#
#   dataset:      midterm | ukr_rus_twitter | covid19_twitter
#   model_list:   path to a text file of checkpoints (see eval_model_list.sbatch)
#   shots_csv:    comma-separated shot counts (default: 0,1,2,5,10)
#
# The eval dataset is determined by <dataset>. For cross-dataset evaluation,
# pass the *target* dataset and a model list from the source training run:
#
#   # Evaluate midterm-trained models on ukr_rus_twitter
#   submit_eval.sh ukr_rus_twitter midterm_train1_eval_on_ukr_rus_model_list.txt

set -euo pipefail

usage() {
  echo "Usage: $0 <dataset> <model_list.txt> [shots_csv]" >&2
  echo "  dataset:   midterm | ukr_rus_twitter | covid19_twitter" >&2
  echo "  shots_csv: e.g. '0,1,2,5,10' (default)" >&2
  exit 1
}

[[ $# -lt 2 || $# -gt 3 ]] && usage

DATASET="$1"; MODEL_LIST="$2"; SHOTS_CSV="${3:-0,1,2,5,10}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/config/${DATASET}.sh"
if [[ ! -f "$CONFIG" ]]; then
  echo "No config found for dataset '${DATASET}'" >&2
  exit 1
fi
source "$CONFIG"

if [[ ! -f "$MODEL_LIST" ]]; then
  echo "Model list not found: $MODEL_LIST" >&2
  exit 1
fi

sbatch \
  --job-name="eval_${DATASET}" \
  --output="${LOG_DIR}/%x_%j.out" \
  --error="${LOG_DIR}/%x_%j.err" \
  --mem="${SLURM_MEM}" \
  --export="ALL,DATASET=${DATASET},MODEL_LIST=${MODEL_LIST},SHOTS_CSV=${SHOTS_CSV}" \
  "${SCRIPT_DIR}/eval_model_list.sbatch"
