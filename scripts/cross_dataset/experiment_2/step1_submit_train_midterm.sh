#!/bin/bash
# Step 1: Pretrain on midterm NM.
# Usage: bash step1_submit_train_midterm.sh
# After job completes: ls state/exp2_train1_midterm_nm_*/state_dict

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting [Exp2] midterm NM training..."
sbatch "${SCRIPT_DIR}/step1_train_midterm.sbatch"
echo "Done. Monitor with: squeue -u \$USER"
