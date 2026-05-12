#!/bin/bash
# Step 1: Pretrain on covid19_twitter LP.
# Usage: bash step1_submit_train_covid.sh
# After job completes: ls state/exp6_train1_covid_lp_*/checkpoint/

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting [Exp6] covid LP training..."
sbatch "${SCRIPT_DIR}/step1_train_covid.sbatch"
echo "Done. Monitor with: squeue -u \$USER"
