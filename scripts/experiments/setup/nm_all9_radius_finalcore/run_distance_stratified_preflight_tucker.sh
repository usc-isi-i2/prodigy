#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT="${OUTPUT:-${REPO_ROOT}/log/nm_all9_distance_stratified/preflight/feasibility.json}"
EPISODES="${EPISODES:-100}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1

available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
required_kib=$((250 * 1024 * 1024))
(( available_kib >= required_kib )) || {
  echo "insufficient host RAM for preflight: require 250 GiB available" >&2
  exit 1
}

mkdir -p "$(dirname "$OUTPUT")"
cd "$REPO_ROOT"
exec "${CONDA_PREFIX}/bin/python" -u \
  "$SCRIPT_DIR/probe_distance_stratified_feasibility.py" \
  --episodes "$EPISODES" --output "$OUTPUT"
