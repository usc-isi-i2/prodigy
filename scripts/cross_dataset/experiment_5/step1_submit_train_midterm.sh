#!/bin/bash
# Step 1: Pretrain on midterm LP.
# Usage: bash step1_submit_train_midterm.sh
# After job completes: ls state/exp5_train1_midterm_lp_*/checkpoint/

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting [Exp5] midterm LP training..."
sbatch "${SCRIPT_DIR}/step1_train_midterm.sbatch"
echo "Done. Monitor with: squeue -u \$USER"
