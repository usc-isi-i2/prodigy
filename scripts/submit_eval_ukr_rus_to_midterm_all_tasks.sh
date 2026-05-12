#!/bin/bash
# Evaluate ukr_rus_twitter-trained models on the midterm graph.
# Usage: submit_eval_ukr_rus_to_midterm_all_tasks.sh [shots_csv]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"${SCRIPT_DIR}/submit_eval.sh" \
  midterm \
  "${SCRIPT_DIR}/ukr_rus_train1_eval_on_midterm_model_list.txt" \
  "${1:-0,1,2,5,10}"
