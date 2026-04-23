#!/bin/bash
# Step 2: Fine-tune ukr_rus NM checkpoint on covid19_twitter NM.
# Usage: bash step2_submit_finetune_covid.sh <ckpt>

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

echo "Submitting [Exp15] fine-tune: ukr_rus NM -> covid NM..."
sbatch --export=ALL,CKPT_PATH="$CKPT_PATH" "${SCRIPT_DIR}/step2_finetune_covid.sbatch"
echo "Done. Monitor with: squeue -u \$USER"
