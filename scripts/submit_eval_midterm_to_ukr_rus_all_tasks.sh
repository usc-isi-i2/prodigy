#!/bin/bash
# Evaluate midterm-trained models on the ukr_rus_twitter graph.
# Usage: submit_eval_midterm_to_ukr_rus_all_tasks.sh [shots_csv]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"${SCRIPT_DIR}/submit_eval.sh" \
  ukr_rus_twitter \
  "${SCRIPT_DIR}/midterm_train1_eval_on_ukr_rus_model_list.txt" \
  "${1:-0,1,2,5,10}"
