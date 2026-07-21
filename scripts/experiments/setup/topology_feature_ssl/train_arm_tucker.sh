#!/usr/bin/env bash
# Train one topology_feature_ssl arm from configs/<ARM>.yaml. Usage:
#   ./train_arm_tucker.sh B0 --device 0
#   ./train_arm_tucker.sh B1 --device 1
#   DRY_RUN=1 ./train_arm_tucker.sh B0 --device 0 --epochs 1 -ds_cap 20 \
#       -eval_step 20 -ckpt_step 20 --prefix tfssl_B0_smoke   # smoke test first
#
# Extra args after the arm name are forwarded to run_single_experiment.py and
# override the config. Run on Tucker in the prodigy env, ideally in tmux.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

[[ $# -lt 1 ]] && { echo "usage: $0 <ARM> [extra args...]  (ARM in configs/*.yaml, e.g. B0)" >&2; exit 2; }
ARM="$1"; shift
CONFIG_PATH="${SCRIPT_DIR}/configs/${ARM}.yaml"
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
log="${SCRIPT_DIR}/run_logs/${ARM}_$(date +%Y%m%d_%H%M%S).log"
echo "logging to ${log}" >&2
"${cmd[@]}" 2>&1 | tee "${log}"
