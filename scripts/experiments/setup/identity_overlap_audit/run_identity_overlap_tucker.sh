#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bio-embeddings-v001
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/dataMeR1/phil/data}"
RUN_TAG="${RUN_TAG:-v001_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/state/identity_overlap_audit/${RUN_TAG}}"

cd "$REPO_ROOT"
python -u scripts/experiments/setup/identity_overlap_audit/audit_identity_overlap.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --memory-limit "${DUCKDB_MEMORY_LIMIT:-80GB}" \
  --threads "${DUCKDB_THREADS:-24}"

echo "Identity-overlap audit complete: $OUTPUT_DIR"
