#!/bin/bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <nm_checkpoint> <lp_checkpoint> <pl_checkpoint>" >&2
  echo "Example: $0 state/train1_ukr_rus_twitter_nm_.../state_dict state/train1_ukr_rus_twitter_lp_.../state_dict state/train1_ukr_rus_twitter_pl_.../state_dict" >&2
  exit 1
fi

cd "$(dirname "$0")"
mkdir -p /scratch1/eibl/data/ukr_rus_twitter/logs

NM_CKPT="$1"
LP_CKPT="$2"
PL_CKPT="$3"

submit_train2() {
  local source_model="$1"
  local ckpt="$2"
  local target_task="$3"
  sbatch \
    --job-name="train2_${source_model}_${target_task}" \
    --export="ALL,SOURCE_MODEL=${source_model},TARGET_TASK=${target_task},CKPT_PATH=${ckpt}" \
    train2_ukr_rus_twitter_single_task.sbatch
}

submit_train2 nm "$NM_CKPT" temporal_link_prediction
submit_train2 nm "$NM_CKPT" classification

submit_train2 lp "$LP_CKPT" neighbor_matching
submit_train2 lp "$LP_CKPT" classification

submit_train2 pl "$PL_CKPT" neighbor_matching
submit_train2 pl "$PL_CKPT" temporal_link_prediction
