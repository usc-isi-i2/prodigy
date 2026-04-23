#!/bin/bash
# Step 3: Evaluate (covid NM -> midterm NM) checkpoint on ukr_rus_twitter (NM + LP + PL, 1,5,10-shot).
# Usage: bash step3_submit_eval_ukr_rus.sh <ckpt>

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
# Exp13: covid NM -> midterm NM -> eval ukr_rus (NM + LP + PL)
covid_nm_to_midterm_nm ${CKPT}
EOF

echo "Model list written to ${MODEL_LIST}:"
cat "$MODEL_LIST"
echo ""

echo "Submitting [Exp13] ukr_rus eval (NM + LP + PL, shots=1,5,10)..."
sbatch "${SCRIPT_DIR}/eval_ukr_rus_twitter_model_list_all_tasks.sbatch" "$MODEL_LIST" "1,5,10"
echo "Done. Monitor with: squeue -u \$USER"
