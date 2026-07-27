#!/usr/bin/env bash
# Train one dense saturation config. Usage:
#   ./train_nm_tucker.sh train_all8_dense.yaml [extra run_single_experiment args...]
# Honors DRY_RUN=1 to print the command without running.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"   # setup/<name> is 4 levels below repo root

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <config.yaml> [extra args...]" >&2
  exit 2
fi

CONFIG_NAME="$1"; shift
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

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"

"${cmd[@]}"
