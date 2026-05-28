#!/bin/bash
set -euo pipefail

SHOTS_CSV="${1:-0,1,2,5,10}"

cd /home1/eibl/gfm/prodigy

sbatch scripts/experiments/eval/eval_midterm_model_list_all_tasks.sbatch \
  scripts/experiments/train1/models/ukr_rus_train1_eval_on_midterm_model_list.txt \
  "$SHOTS_CSV"
