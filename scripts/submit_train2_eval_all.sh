#!/bin/bash
# Evaluate all train2 models on all 6 graphs.
# lp is skipped on ukr_rus_suspended, covid_political, election2020 (not supported).
#
# Usage: bash scripts/submit_train2_eval_all.sh [model_list.txt] [shots_csv]

set -euo pipefail

MODEL_LIST="${1:-scripts/model_lists/train2_all_models.txt}"
SHOTS_CSV="${2:-0,3}"

cd /home1/eibl/gfm/prodigy

sbatch scripts/eval_ukr_rus_suspended_model_list_all_tasks.sbatch  "$MODEL_LIST" "$SHOTS_CSV"
sbatch scripts/eval_covid19_twitter_model_list_all_tasks.sbatch    "$MODEL_LIST" "$SHOTS_CSV"
sbatch scripts/eval_covid_political_model_list_all_tasks.sbatch    "$MODEL_LIST" "$SHOTS_CSV"
sbatch scripts/eval_midterm_model_list_all_tasks.sbatch            "$MODEL_LIST" "$SHOTS_CSV"
sbatch scripts/eval_election2020_model_list_all_tasks.sbatch       "$MODEL_LIST" "$SHOTS_CSV"
sbatch scripts/eval_ukr_rus_twitter_model_list_all_tasks.sbatch    "$MODEL_LIST" "$SHOTS_CSV"
