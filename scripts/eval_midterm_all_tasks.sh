#!/bin/bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <nm_checkpoint> <lp_checkpoint> <pl_checkpoint>" >&2
  echo "Example: $0 state/pretrain_midterm_nm_.../state_dict state/midterm_lp_sanity_.../state_dict state/midterm_pl_sanity_emb_only_.../state_dict" >&2
  exit 1
fi

cd "$(dirname "$0")"

NM_CKPT="$1"
LP_CKPT="$2"
PL_CKPT="$3"

submit_eval() {
  local model_name="$1"
  local ckpt="$2"
  local task_name="$3"
  sbatch \
    --job-name="eval_${model_name}_${task_name}" \
    --export="ALL,MODEL_NAME=${model_name},TASK_NAME=${task_name},CKPT_PATH=${ckpt}" \
    eval_midterm_single_task.sbatch
}

for task_name in neighbor_matching temporal_link_prediction classification; do
  submit_eval nm "$NM_CKPT" "$task_name"
  submit_eval lp "$LP_CKPT" "$task_name"
  submit_eval pl "$PL_CKPT" "$task_name"
done
