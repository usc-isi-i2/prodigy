#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODEL_LIST="${MODEL_LIST:-${SCRIPT_DIR}/merged_ukr_rus_covid_midterm_nm_aug_transfer_trajectory_model_list.txt}"
DATA_ROOT="${DATA_ROOT:-/dataMeR1/phil/data}"
DATASETS="${DATASETS:-midterm,covid19_twitter,ukr_rus_twitter,covid_political,election2020,ukr_rus_suspended}"
TASKS="${TASKS:-nm,lp,pl}"
SHOTS="${SHOTS:-0,3,10}"
GPUS="${GPUS:-}"
MODEL_DROPOUT="${MODEL_DROPOUT:-0.1}"

cmd=(
  python3 scripts/experiments/eval/eval_ckpts_all_graph_tasks_tucker.py
  --model-list "${MODEL_LIST}"
  --data-root "${DATA_ROOT}"
  --datasets "${DATASETS}"
  --tasks "${TASKS}"
  --shots "${SHOTS}"
)

if [[ -n "${GPUS}" ]]; then
  cmd+=(--gpus "${GPUS}")
fi

cmd+=("$@")
cmd+=(
  --
  --dropout "${MODEL_DROPOUT}"
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
