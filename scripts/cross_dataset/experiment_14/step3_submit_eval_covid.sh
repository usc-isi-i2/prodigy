#!/bin/bash
# Step 3: Evaluate (ukr_rus NM -> midterm NM) checkpoint on covid19_twitter (NM + LP + PL, 1,5,10-shot).
# Usage: bash step3_submit_eval_covid.sh <ckpt>

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
MODEL_LIST="${SCRIPT_DIR}/step3_covid_model_list.txt"

cat > "$MODEL_LIST" <<EOF
# Exp14: ukr_rus NM -> midterm NM -> eval covid (NM + LP + PL)
ukr_rus_nm_to_midterm_nm ${CKPT}
EOF

echo "Model list written to ${MODEL_LIST}:"
cat "$MODEL_LIST"
echo ""

echo "Submitting [Exp14] covid eval (NM + LP + PL, shots=1,5,10)..."
sbatch "${SCRIPT_DIR}/eval_covid19_twitter_model_list_all_tasks.sbatch" "$MODEL_LIST" "1,5,10"
echo "Done. Monitor with: squeue -u \$USER"
