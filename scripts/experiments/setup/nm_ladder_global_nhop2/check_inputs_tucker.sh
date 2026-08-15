#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/dataMeR1/phil/data}"
python3 "${SCRIPT_DIR}/make_configs.py" --check
tail -n +2 "${SCRIPT_DIR}/manifest.tsv" | while IFS=$'\t' read -r _rung _prefix _dataset _graph_id filename; do
  [[ -s "${DATA_ROOT}/merged/graphs/${filename}" ]] || {
    echo "missing merged artifact: ${DATA_ROOT}/merged/graphs/${filename}" >&2
    exit 1
  }
done
echo "OK: configs and all seven growing-merge artifacts are present"
