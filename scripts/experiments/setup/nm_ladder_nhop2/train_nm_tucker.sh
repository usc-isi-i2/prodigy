#!/usr/bin/env bash
# Train one generated 2-hop ladder config on Tucker.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <config path relative to this folder> [extra trainer args...]" >&2
  exit 2
fi

CONFIG_NAME="$1"
shift
CONFIG_PATH="${SCRIPT_DIR}/${CONFIG_NAME}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

cmd=(
  python3 experiments/run_single_experiment.py
  --config "${CONFIG_PATH}"
  "$@"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

export PATH="/home/mhchu/miniconda3/bin:${PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"
"${cmd[@]}"
