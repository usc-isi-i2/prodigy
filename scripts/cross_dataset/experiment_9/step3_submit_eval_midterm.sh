#!/bin/bash
# Step 3: Evaluate (covid NM -> ukr_rus LP) checkpoint on midterm (NM + LP + PL, 1,5,10-shot).
# Usage: bash step3_submit_eval_midterm.sh <ckpt>

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
MODEL_LIST="${SCRIPT_DIR}/step3_midterm_model_list.txt"

cat > "$MODEL_LIST" <<EOF
# Exp9: covid NM -> ukr_rus LP -> eval midterm (NM + LP + PL)
covid_nm_to_ukr_rus_lp ${CKPT}
EOF

echo "Model list written to ${MODEL_LIST}:"
cat "$MODEL_LIST"
echo ""

echo "Submitting [Exp9] midterm eval (NM + LP + PL, shots=1,5,10)..."
sbatch "${SCRIPT_DIR}/eval_midterm_model_list_all_tasks.sbatch" "$MODEL_LIST" "1,5,10"
echo "Done. Monitor with: squeue -u \$USER"
