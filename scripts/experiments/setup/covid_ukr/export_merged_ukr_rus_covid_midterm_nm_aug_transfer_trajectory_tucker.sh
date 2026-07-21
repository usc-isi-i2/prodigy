#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/scripts/experiments/analysis/archive/transfer_trajectory_merged_ukr_rus_covid_midterm_nm_aug}"
RUN_GLOB="${RUN_GLOB:-eval_merged_ukr_rus_covid_midterm_nm_aug_15_06_2026_17_49_22_step*_to_*}"

if [[ -z "${LOG_ROOT:-}" ]]; then
  for candidate in \
    /dataMeR1/phil/gfm/prodigy/log \
    /dataMeR2/phil/gfm/prodigy/log \
    "${REPO_ROOT}/log"
  do
    matches=("${candidate}"/${RUN_GLOB})
    if [[ -e "${matches[0]}" ]]; then
      LOG_ROOT="${candidate}"
      break
    fi
  done
fi

LOG_ROOT="${LOG_ROOT:-/dataMeR1/phil/gfm/prodigy/log}"

cmd=(
  python3 scripts/harness/export_eval_results_csv.py
  --log-root "${LOG_ROOT}"
  --run-glob "${RUN_GLOB}"
  --out-dir "${OUT_DIR}"
  --datasets midterm,covid19_twitter,ukr_rus_twitter,covid_political,election2020,ukr_rus_suspended
  --tasks nm,lp,pl
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
