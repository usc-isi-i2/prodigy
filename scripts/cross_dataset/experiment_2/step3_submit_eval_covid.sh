#!/bin/bash
# Step 3: Evaluate (midterm NM -> ukr_rus NM) checkpoint on covid19_twitter (NM + LP + PL, 1,5,10-shot).
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
# Exp2: midterm NM -> ukr_rus NM -> eval covid19_twitter (NM + LP + PL)
midterm_nm_to_ukr_rus_nm ${CKPT}
EOF

echo "Model list written to ${MODEL_LIST}:"
cat "$MODEL_LIST"
echo ""

echo "Submitting [Exp2] covid eval (NM + LP, 1,5,10-shot)..."
sbatch "${SCRIPT_DIR}/eval_covid19_twitter_model_list_all_tasks.sbatch" "$MODEL_LIST" "1,5,10"

echo "Submitting [Exp2] covid eval (PL, 1,5,10-shot)..."
sbatch "${SCRIPT_DIR}/eval_covid19_twitter_pl.sbatch" "$MODEL_LIST" "1,5,10"

echo "Done. Monitor with: squeue -u \$USER"
