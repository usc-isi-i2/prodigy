#!/bin/bash
set -euo pipefail

SHOTS_CSV="${1:-0,1,3,10}"
TAGS_CSV="${2:-}"
MODEL_LIST="scripts/model_lists/eval_covid_political_pl.txt"

cd /home1/eibl/gfm/prodigy
mkdir -p logs

SBATCH_ARGS=("$MODEL_LIST" "$SHOTS_CSV")
if [[ -n "$TAGS_CSV" ]]; then
  SBATCH_ARGS+=("$TAGS_CSV")
fi

sbatch scripts/eval_midterm_model_list_all_tasks.sbatch "${SBATCH_ARGS[@]}"
sbatch scripts/eval_covid19_twitter_model_list_all_tasks.sbatch "${SBATCH_ARGS[@]}"
sbatch scripts/eval_ukr_rus_twitter_model_list_all_tasks.sbatch "${SBATCH_ARGS[@]}"
