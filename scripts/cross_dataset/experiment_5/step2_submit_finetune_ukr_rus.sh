#!/bin/bash
# Step 2: Fine-tune midterm LP checkpoint on ukr_rus_twitter LP.
# Usage: bash step2_submit_finetune_ukr_rus.sh <ckpt>

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

echo "Submitting [Exp5] fine-tune: midterm LP -> ukr_rus LP..."
sbatch --export=ALL,CKPT_PATH="$CKPT_PATH" "${SCRIPT_DIR}/step2_finetune_ukr_rus.sbatch"
echo "Done. Monitor with: squeue -u \$USER"
