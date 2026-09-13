#!/usr/bin/env bash
# Wait for the audited flagship grids, then run the preregistered analysis once.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_STAMP="${RUN_STAMP:-20260908}"
SOURCE_ROOT="${SOURCE_ROOT:-/dataMeR1/phil/gfm/prodigy-paper-remainder-opt/log/paper_flagship_ladders/${RUN_STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/scripts/experiments/analysis/synthesis/cross_experiment/paper_flagship_ladders}"
POLL_SECONDS="${POLL_SECONDS:-60}"
NM_STATUS="${SOURCE_ROOT}/nm_evaluation/status.json"
NM_CELLS="${SOURCE_ROOT}/nm_evaluation/cells"
SEED0_NM_STATUS="${SOURCE_ROOT}/nm_evaluation_seed0_refresh/status.json"
SEED0_NM_CELLS="${SOURCE_ROOT}/nm_evaluation_seed0_refresh/cells"
CLS_ROOT="${SOURCE_ROOT}/classification_evaluation"
CLS_TABLE="${CLS_ROOT}/classification_long.tsv"
CLS_COMPLETE="${CLS_ROOT}/complete_utc.txt"
STATUS_FILE="${OUTPUT_ROOT}/postprocess_status.json"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${REPO_ROOT}/log/matplotlib-cache}"
mkdir -p "$OUTPUT_ROOT" "$MPLCONFIGDIR"
cd "$REPO_ROOT"

write_status() {
  local status="$1"
  local detail="$2"
  "${CONDA_PREFIX}/bin/python" - "$STATUS_FILE" "$status" "$detail" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "detail": sys.argv[3],
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}
path.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
    temporary = handle.name
os.replace(temporary, path)
PY
}

trap 'write_status failed "flagship postprocessing failed"' ERR
write_status waiting "waiting for 864 replica NM, 432 refreshed seed-0 NM, and 600 classification cells"

while true; do
  if [[ -f "$NM_STATUS" ]] && grep -q '"status": "failed"' "$NM_STATUS"; then
    echo "flagship NM evaluation failed" >&2
    exit 1
  fi
  if [[ -f "$SEED0_NM_STATUS" ]] && grep -q '"status": "failed"' "$SEED0_NM_STATUS"; then
    echo "flagship seed-0 NM reevaluation failed" >&2
    exit 1
  fi
  if [[ -f "$NM_STATUS" ]] && grep -q '"status": "complete"' "$NM_STATUS" \
      && [[ -f "$SEED0_NM_STATUS" ]] && grep -q '"status": "complete"' "$SEED0_NM_STATUS" \
      && [[ -f "$CLS_COMPLETE" && -f "$CLS_TABLE" ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

"${CONDA_PREFIX}/bin/python" - "$NM_CELLS" "$SEED0_NM_CELLS" "$CLS_TABLE" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

nm_root = Path(sys.argv[1])
seed0_nm_root = Path(sys.argv[2])
cls_path = Path(sys.argv[3])
nm_paths = sorted(nm_root.glob("*/*.json"))
if len(nm_paths) != 864:
    raise ValueError(f"expected 864 flagship NM cells, found {len(nm_paths)}")
for path in nm_paths:
    row = json.loads(path.read_text(encoding="utf-8"))
    if int(row.get("episodes", -1)) != 512 or not math.isfinite(float(row["roc_auc"])):
        raise ValueError(f"invalid flagship NM cell: {path}")
seed0_nm_paths = sorted(seed0_nm_root.glob("*/*.json"))
if len(seed0_nm_paths) != 432:
    raise ValueError(f"expected 432 refreshed seed-0 NM cells, found {len(seed0_nm_paths)}")
for path in seed0_nm_paths:
    row = json.loads(path.read_text(encoding="utf-8"))
    if int(row.get("seed", -1)) != 0 or int(row.get("episodes", -1)) != 512 \
            or row.get("legacy_test_only") is not True \
            or not math.isfinite(float(row["roc_auc"])):
        raise ValueError(f"invalid refreshed seed-0 NM cell: {path}")
with cls_path.open(encoding="utf-8", newline="") as handle:
    cls_rows = list(csv.DictReader(handle, delimiter="\t"))
if len(cls_rows) != 600:
    raise ValueError(f"expected 600 flagship classification cells, found {len(cls_rows)}")
PY

write_status analyzing "input cardinalities passed; running preregistered analysis"
"${CONDA_PREFIX}/bin/python" \
  scripts/experiments/analysis/synthesis/cross_experiment/paper_flagship_ladders/analyze.py \
  --seed0-nm-root "$SEED0_NM_CELLS" \
  --replicate-nm-root "$NM_CELLS" \
  --classification "$CLS_TABLE" \
  --output-root "$OUTPUT_ROOT"
write_status complete "flagship analysis and figures complete"
trap - ERR
