#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT="${OUTPUT:-${REPO_ROOT}/log/nm_all9_radius_finalcore/preflight/feasibility.json}"
EPISODES="${EPISODES:-100}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"

cd "$REPO_ROOT"
mkdir -p "$(dirname "$OUTPUT")"
"$PYTHON" -u "$SCRIPT_DIR/probe_radius_feasibility.py" \
  --config "$SCRIPT_DIR/radius_mix.yaml" \
  --episodes "$EPISODES" \
  --seed 0 \
  --output "$OUTPUT"

echo "READY radius feasibility gate passed: $OUTPUT"
