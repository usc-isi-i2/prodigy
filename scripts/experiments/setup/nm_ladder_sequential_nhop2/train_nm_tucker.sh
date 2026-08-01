#!/usr/bin/env bash
# Train one sequential-ladder config. Honors DRY_RUN=1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <config.yaml> [run_single_experiment overrides...]" >&2
  exit 2
fi

CONFIG="$1"; shift
if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${SCRIPT_DIR}/configs/${CONFIG}"
fi
[[ -f "${CONFIG}" ]] || { echo "missing config: ${CONFIG}" >&2; exit 2; }

cmd=(python3 experiments/run_single_experiment.py --config "${CONFIG}" "$@")
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
