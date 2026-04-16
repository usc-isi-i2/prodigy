#!/bin/bash
# Step 3: Evaluate the (midterm NM -> covid NM) checkpoint on ukr_rus_twitter.
# Runs all 3 tasks (NM, LP, PL) at 10 shots.
# Run after step2 job completes and you have the fine-tuned checkpoint path.
#
# Usage: bash step3_submit_eval_ukr_rus.sh <ckpt>
#
# Example:
#   bash step3_submit_eval_ukr_rus.sh \
#     ../state/train2_midterm_nm_to_covid_nm_<run>/state_dict

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <ckpt>" >&2
  exit 1
fi

CKPT="$1"

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_LIST="${SCRIPT_DIR}/step3_ukr_rus_model_list.txt"

cat > "$MODEL_LIST" <<EOF
# Sequential cross-dataset: midterm NM -> covid NM -> eval ukr_rus (NM + LP + PL)
# Format: <model_name> <checkpoint_path>
midterm_nm_to_covid_nm ${CKPT}
EOF

echo "Model list written to ${MODEL_LIST}:"
cat "$MODEL_LIST"
echo ""

echo "Submitting ukr_rus eval (NM + LP + PL, 10-shot)..."
sbatch "${SCRIPT_DIR}/eval_ukr_rus_twitter_model_list_all_tasks.sbatch" \
  "$MODEL_LIST" \
  "10"

echo "Done. Monitor with: squeue -u \$USER"
