#!/usr/bin/env bash
# Train the within-source-episode merged model. Default GPU 3 (free during the
# nm_transfer_matrix runs on 0/1/2). Override with --device N. DRY_RUN=1 to preview.
#   ./train_tucker.sh                 # GPU 3
#   ./train_tucker.sh --device 5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/merged_within_source_nm.yaml"

# default device 3 unless caller passes --device
extra=("$@")
has_device=0
for a in "$@"; do [[ "$a" == "--device" ]] && has_device=1; done
[[ $has_device -eq 0 ]] && extra=(--device 3 "$@")

cmd=(python3 experiments/run_single_experiment.py --config "${CONFIG_PATH}" "${extra[@]}")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY:'; printf ' %q' "${cmd[@]}"; printf '\n'; exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"
mkdir -p "${SCRIPT_DIR}/run_logs"
log="${SCRIPT_DIR}/run_logs/within_source_$(date +%Y%m%d_%H%M%S).log"
echo "logging to ${log}" >&2
"${cmd[@]}" 2>&1 | tee "${log}"
