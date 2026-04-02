#!/bin/bash
set -euo pipefail

SHOTS_CSV="${1:-1,2,5,10}"

cd /home1/eibl/gfm/prodigy

sbatch scripts/eval_ukr_rus_twitter_model_list_all_tasks.sbatch \
  scripts/midterm_train1_eval_on_ukr_rus_model_list.txt \
  "$SHOTS_CSV"
