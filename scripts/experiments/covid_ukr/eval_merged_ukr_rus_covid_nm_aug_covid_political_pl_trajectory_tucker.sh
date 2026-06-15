#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

RUN_DIR="${RUN_DIR:-/dataMeR2/phil/gfm/prodigy/state/merged_ukr_rus_covid_nm_aug_15_06_2026_15_22_07}"

cmd=(
  python3 scripts/experiments/eval/eval_ckpts_all_graph_tasks_tucker.py
  --checkpoint-run-dir "${RUN_DIR}"
  --datasets covid_political
  --tasks pl
  --shots 3
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
