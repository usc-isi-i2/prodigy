#!/bin/bash
# Step 3: Evaluate (covid NM -> ukr_rus NM) checkpoint on midterm (NM + LP + PL, 10-shot).
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
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODEL_LIST="${SCRIPT_DIR}/step3_midterm_model_list.txt"

cat > "$MODEL_LIST" <<EOF
# Exp3: covid NM -> ukr_rus NM -> eval midterm (NM + LP + PL)
covid_nm_to_ukr_rus_nm ${CKPT}
EOF

echo "Model list written to ${MODEL_LIST}:"
cat "$MODEL_LIST"
echo ""

echo "Submitting [Exp3] midterm eval (NM + LP + PL, 10-shot)..."
sbatch "${REPO_ROOT}/eval_midterm_model_list_all_tasks.sbatch" "$MODEL_LIST" "1,5,10"
echo "Done. Monitor with: squeue -u \$USER"
