#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

[[ $# -ge 1 ]] || { echo "usage: $0 <config relative to this folder> [trainer args...]" >&2; exit 2; }
CONFIG_PATH="${SCRIPT_DIR}/$1"
shift
[[ -f "${CONFIG_PATH}" ]] || { echo "missing config: ${CONFIG_PATH}" >&2; exit 2; }

cmd=(python3 experiments/run_single_experiment.py --config "${CONFIG_PATH}" "$@")
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
