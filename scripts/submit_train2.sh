#!/bin/bash
# Submit all train2 (cross-task fine-tuning) jobs for a dataset.
# Each train1 checkpoint is fine-tuned on the two tasks it was not trained on.
#
# Usage: submit_train2.sh <dataset> <nm_ckpt> <lp_ckpt> <pl_ckpt>
#
#   dataset:  midterm | ukr_rus_twitter | covid19_twitter
#   nm_ckpt:  checkpoint from train1 neighbor_matching
#   lp_ckpt:  checkpoint from train1 temporal_link_prediction
#   pl_ckpt:  checkpoint from train1 classification (pass "" if unsupported)
#
# Example:
#   submit_train2.sh midterm \
#     state/train1_midterm_nm_.../state_dict \
#     state/train1_midterm_lp_.../state_dict \
#     state/train1_midterm_pl_.../state_dict

set -euo pipefail

usage() {
  echo "Usage: $0 <dataset> <nm_ckpt> <lp_ckpt> <pl_ckpt>" >&2
  exit 1
}

[[ $# -ne 4 ]] && usage

DATASET="$1"; NM_CKPT="$2"; LP_CKPT="$3"; PL_CKPT="$4"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/config/${DATASET}.sh"
if [[ ! -f "$CONFIG" ]]; then
  echo "No config found for dataset '${DATASET}'" >&2
  exit 1
fi
source "$CONFIG"

submit() {
  local src="$1"   # nm | lp | pl
  local ckpt="$2"
  local target="$3"  # nm | lp | pl

  if [[ -z "$ckpt" ]]; then
    echo "Skipping train2 ${src}→${target}: no checkpoint provided." >&2
    return
  fi

  if [[ ! -f "$ckpt" ]]; then
    echo "Checkpoint not found for ${src}: ${ckpt}" >&2
    exit 1
  fi

  local job_name="train2_${DATASET}_${src}_to_${target}"

  sbatch \
    --job-name="$job_name" \
    --output="${LOG_DIR}/%x_%j.out" \
    --error="${LOG_DIR}/%x_%j.err" \
    --mem="${SLURM_MEM}" \
    --export="ALL,DATASET=${DATASET},TASK=${target},CKPT_PATH=${ckpt},PREFIX_OVERRIDE=${job_name}" \
    "${SCRIPT_DIR}/train_single_task.sbatch"
}

# Each source model is fine-tuned on the two tasks it was not trained on.
submit nm "$NM_CKPT" lp
submit nm "$NM_CKPT" pl

submit lp "$LP_CKPT" nm
submit lp "$LP_CKPT" pl

submit pl "$PL_CKPT" nm
submit pl "$PL_CKPT" lp
