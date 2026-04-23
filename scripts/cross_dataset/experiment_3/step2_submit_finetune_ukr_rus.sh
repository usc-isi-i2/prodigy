#!/bin/bash
# Step 2: Fine-tune covid NM checkpoint on ukr_rus_twitter NM.
# Usage: bash step2_submit_finetune_ukr_rus.sh <ckpt>
# After job completes: ls state/exp3_train2_covid_nm_to_ukr_rus_nm_*/checkpoint/

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <ckpt>" >&2
  exit 1
fi

CKPT_PATH="$1"
if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Checkpoint not found: $CKPT_PATH" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting [Exp3] fine-tune: covid NM -> ukr_rus NM..."
sbatch --export=ALL,CKPT_PATH="$CKPT_PATH" "${SCRIPT_DIR}/step2_finetune_ukr_rus.sbatch"
echo "Done. Monitor with: squeue -u \$USER"
