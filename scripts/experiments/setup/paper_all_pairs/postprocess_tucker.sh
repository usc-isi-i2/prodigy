#!/usr/bin/env bash
# Wait for the exact pair evaluator audit, then run the preregistered analysis.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/dataMeR1/phil/gfm/prodigy-paper-all-pairs/log/paper_all_pairs/20260908}"
EVAL_ROOT="${RUN_ROOT}/nm_evaluation"
OUTPUT="${RUN_ROOT}/analysis"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
cd "$REPO_ROOT"

while ! "${CONDA_PREFIX}/bin/python" - "$EVAL_ROOT/status.json" <<'PY' 2>/dev/null
import json
import sys
status = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if status.get("status") == "complete" and status.get("audit", {}).get("cells") == 972 else 1)
PY
do
  sleep 60
done

"${CONDA_PREFIX}/bin/python" -u \
  scripts/experiments/analysis/synthesis/cross_experiment/paper_all_pairs/analyze.py \
  --pair-eval "$EVAL_ROOT" --output "$OUTPUT"
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUTPUT/complete_utc.txt"
