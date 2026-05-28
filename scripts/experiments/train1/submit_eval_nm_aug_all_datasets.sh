#!/bin/bash
set -euo pipefail

SHOTS_CSV="${1:-0,3,10}"
TAGS_CSV="${2:-aug}"
MODEL_LIST="scripts/experiments/eval/models/eval_nm_aug.txt"

cd /home1/eibl/gfm/prodigy
mkdir -p logs

sbatch scripts/experiments/eval/eval_midterm_model_list_all_tasks.sbatch           "$MODEL_LIST" "$SHOTS_CSV" "$TAGS_CSV"
sbatch scripts/experiments/eval/eval_covid19_twitter_model_list_all_tasks.sbatch   "$MODEL_LIST" "$SHOTS_CSV" "$TAGS_CSV"
sbatch scripts/experiments/eval/eval_ukr_rus_suspended_model_list_all_tasks.sbatch "$MODEL_LIST" "$SHOTS_CSV" "$TAGS_CSV"
