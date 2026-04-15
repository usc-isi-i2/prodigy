#!/bin/bash
# Step 2: Fine-tune midterm NM checkpoint on covid19_twitter NM.
# Run after step1 job completes and you have the checkpoint path.
#
# Usage: bash step2_submit_finetune_covid.sh <nm_ckpt>
#
# Example:
#   bash step2_submit_finetune_covid.sh \
#     ../state/train1_midterm_nm_<run>/state_dict
#
# After job completes, find the checkpoint with:
#   ls ../state/train2_midterm_nm_to_covid_nm_*/state_dict

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <nm_ckpt>" >&2
  exit 1
fi

CKPT_PATH="$1"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Checkpoint not found: $CKPT_PATH" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p /scratch1/eibl/data/covid19_twitter/logs

echo "Submitting fine-tune: midterm NM -> covid NM..."
sbatch \
  --export=ALL,CKPT_PATH="$CKPT_PATH" \
  "${SCRIPT_DIR}/step2_finetune_covid.sbatch"

echo "Done. Monitor with: squeue -u \$USER"
echo ""
echo "After job completes, find checkpoint with:"
echo "  ls ../state/train2_midterm_nm_to_covid_nm_*/state_dict"
