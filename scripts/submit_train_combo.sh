#!/bin/bash
# Submit a single combo experiment: fine-tune a train1 checkpoint on a second
# task, then evaluate on the held-out third task.
#
# Usage:
#   submit_train_combo.sh <dataset> <src_task> <src_ckpt> <finetune_task> <eval_task> [shots_csv]
#
#   dataset:       midterm | ukr_rus_twitter | covid19_twitter
#   src_task:      task the source checkpoint was trained on  (nm | lp | pl)
#   src_ckpt:      path to the train1 checkpoint
#   finetune_task: task to fine-tune on                      (nm | lp | pl)
#   eval_task:     held-out task to evaluate on              (nm | lp | pl)
#   shots_csv:     comma-separated shot counts (default: 0,1,2,5,10)
#
# All three tasks must be distinct.
#
# Example:
#   submit_train_combo.sh midterm nm state/train1_midterm_nm_.../state_dict lp pl

set -euo pipefail

usage() {
  echo "Usage: $0 <dataset> <src_task> <src_ckpt> <finetune_task> <eval_task> [shots_csv]" >&2
  exit 1
}

[[ $# -lt 5 || $# -gt 6 ]] && usage

DATASET="$1"
SRC_TASK="$2"
SRC_CKPT="$3"
FINETUNE_TASK="$4"
EVAL_TASK="$5"
SHOTS_CSV="${6:-0,1,2,5,10}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/config/${DATASET}.sh"
if [[ ! -f "$CONFIG" ]]; then
  echo "No config found for dataset '${DATASET}'" >&2
  exit 1
fi
source "$CONFIG"

if [[ ! -f "$SRC_CKPT" ]]; then
  echo "Source checkpoint not found: $SRC_CKPT" >&2
  exit 1
fi

JOB_NAME="combo_${DATASET}_${SRC_TASK}_ft_${FINETUNE_TASK}_eval_${EVAL_TASK}"

sbatch \
  --job-name="$JOB_NAME" \
  --output="${LOG_DIR}/%x_%j.out" \
  --error="${LOG_DIR}/%x_%j.err" \
  --mem="${SLURM_MEM}" \
  --export="ALL,\
DATASET=${DATASET},\
SRC_TASK=${SRC_TASK},\
SRC_CKPT=${SRC_CKPT},\
FINETUNE_TASK=${FINETUNE_TASK},\
EVAL_TASK=${EVAL_TASK},\
SHOTS_CSV=${SHOTS_CSV}" \
  "${SCRIPT_DIR}/train_combo.sbatch"
