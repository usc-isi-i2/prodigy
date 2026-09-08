#!/usr/bin/env bash
# Freeze the complete mechanism analysis, including long-form NM and CLS cells,
# from a revision isolated from the live training/evaluation worktree.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_STAMP="${RUN_STAMP:-20260908}"
RUN_ROOT="${RUN_ROOT:-/dataMeR1/phil/gfm/prodigy-paper-mechanism/log/paper_mechanism_sweeps/${RUN_STAMP}}"
REVISION="$(git -C "$REPO_ROOT" rev-parse HEAD)"
OUTPUT="${OUTPUT:-${RUN_ROOT}/analysis_frozen_${REVISION:0:8}}"

while [[ ! -f "$RUN_ROOT/complete_utc.txt" ]]; do sleep 60; done

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
cd "$REPO_ROOT"

if [[ -e "$OUTPUT" ]]; then
  "${CONDA_PREFIX}/bin/python" - "$OUTPUT" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
row = json.load((root / "data/audit.json").open(encoding="utf-8"))
expected = {"status": "complete", "nm_cells": 810, "classification_cells": 375,
            "physical_checkpoint_models": 90, "cross_task_models": 75}
for key, value in expected.items():
    if row.get(key) != value:
        raise ValueError(f"existing frozen analysis failed {key}: {row.get(key)!r}")
for relative, count in (("data/nm_cells.csv", 810),
                        ("data/classification_cells.csv", 375),
                        ("data/cross_task_model_provenance.csv", 75)):
    with (root / relative).open(encoding="utf-8", newline="") as handle:
        observed = sum(1 for _ in csv.reader(handle)) - 1
    if observed != count:
        raise ValueError(f"{relative}: expected {count}, found {observed}")
PY
  exit 0
fi

"${CONDA_PREFIX}/bin/python" \
  scripts/experiments/analysis/synthesis/cross_experiment/paper_mechanism_sweeps/analyze.py \
  --nm-root "$RUN_ROOT/nm_evaluation" \
  --classification "$RUN_ROOT/classification_evaluation/classification_long.tsv" \
  --output "$OUTPUT"

"${CONDA_PREFIX}/bin/python" - "$OUTPUT" "$REVISION" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
audit = json.load((root / "data/audit.json").open(encoding="utf-8"))
expected = {"status": "complete", "nm_cells": 810, "classification_cells": 375,
            "physical_checkpoint_models": 90, "cross_task_models": 75}
for key, value in expected.items():
    if audit.get(key) != value:
        raise ValueError(f"frozen analysis failed {key}: {audit.get(key)!r}")
for relative, count in (("data/nm_cells.csv", 810),
                        ("data/classification_cells.csv", 375),
                        ("data/cross_task_model_provenance.csv", 75)):
    with (root / relative).open(encoding="utf-8", newline="") as handle:
        observed = sum(1 for _ in csv.reader(handle)) - 1
    if observed != count:
        raise ValueError(f"{relative}: expected {count}, found {observed}")
(root / "analysis_revision.txt").write_text(sys.argv[2] + "\n", encoding="utf-8")
PY
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUTPUT/complete_utc.txt"
