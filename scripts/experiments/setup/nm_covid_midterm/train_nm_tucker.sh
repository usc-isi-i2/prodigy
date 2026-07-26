#!/usr/bin/env bash
# Train one cov/mid regime. Usage:
#   ./train_nm_tucker.sh midterm_nm.yaml --device 0
#   DRY_RUN=1 ./train_nm_tucker.sh merged_within_balanced_nm.yaml --device 3
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

[[ $# -lt 1 ]] && { echo "usage: $0 <config.yaml> [extra args...]" >&2; exit 2; }
CONFIG_NAME="$1"; shift
CONFIG_PATH="${SCRIPT_DIR}/${CONFIG_NAME}"
[[ -f "${CONFIG_PATH}" ]] || { echo "config not found: ${CONFIG_PATH}" >&2; exit 2; }

cmd=(python3 experiments/run_single_experiment.py --config "${CONFIG_PATH}" "$@")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY:'; printf ' %q' "${cmd[@]}"; printf '\n'; exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"
mkdir -p "${SCRIPT_DIR}/run_logs"
log="${SCRIPT_DIR}/run_logs/${CONFIG_NAME%.yaml}_$(date +%Y%m%d_%H%M%S).log"
echo "logging to ${log}" >&2
"${cmd[@]}" 2>&1 | tee "${log}"
