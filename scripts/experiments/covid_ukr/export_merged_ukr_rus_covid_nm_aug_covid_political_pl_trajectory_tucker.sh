#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

LOG_ROOT="${LOG_ROOT:-/dataMeR2/phil/gfm/prodigy/log}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/scripts/plotting/covid_political_pl_trajectory_aug}"
RUN_GLOB="${RUN_GLOB:-eval_merged_ukr_rus_covid_nm_aug_15_06_2026_15_22_07_step*_to_covid_political_pl_3shot_*}"

cmd=(
  python3 scripts/analysis/export_eval_results_csv.py
  --log-root "${LOG_ROOT}"
  --run-glob "${RUN_GLOB}"
  --out-dir "${OUT_DIR}"
  --datasets covid_political
  --tasks pl
  --splits test
  --duplicate-policy latest
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
