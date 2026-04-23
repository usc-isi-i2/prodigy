#!/bin/bash
# Step 1: Pretrain on ukr_rus_twitter NM.
# Usage: bash step1_submit_train_ukr_rus.sh
# After job completes: ls state/exp14_train1_ukr_rus_nm_*/checkpoint/
# Note: this checkpoint is also reused by Experiment 15.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting [Exp14] ukr_rus NM training..."
sbatch "${SCRIPT_DIR}/step1_train_ukr_rus.sbatch"
echo "Done. Monitor with: squeue -u \$USER"
