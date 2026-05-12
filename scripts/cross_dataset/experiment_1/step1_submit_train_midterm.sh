#!/bin/bash
# Step 1: Pretrain on midterm NM.
# Output checkpoint is the input to step2_submit_finetune_covid.sh.
#
# Usage: bash step1_submit_train_midterm.sh
#
# After job completes, find the checkpoint with:
#   ls state/train1_midterm_nm_*/state_dict

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting midterm NM training..."
sbatch "${SCRIPT_DIR}/step1_train_midterm.sbatch"

echo "Done. Monitor with: squeue -u \$USER"
