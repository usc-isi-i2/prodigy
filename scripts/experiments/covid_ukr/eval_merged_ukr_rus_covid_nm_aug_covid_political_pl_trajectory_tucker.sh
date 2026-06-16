#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

RUN_DIR="${RUN_DIR:-/dataMeR1/phil/gfm/prodigy/state/merged_ukr_rus_covid_nm_aug_15_06_2026_15_22_07}"
DATA_ROOT="${DATA_ROOT:-/dataMeR1/phil/data}"
DATASETS="${DATASETS:-covid_political}"
TASKS="${TASKS:-pl}"
SHOTS="${SHOTS:-3}"
GPUS="${GPUS:-}"
CHECKPOINT_NAME_PREFIX="${CHECKPOINT_NAME_PREFIX:-}"
EMB_DIM="${EMB_DIM:-512}"
LAYERS="${LAYERS:-S2,U,M2}"
MODEL_DROPOUT="${MODEL_DROPOUT:-0.1}"

cmd=(
  python3 scripts/experiments/eval/eval_ckpts_all_graph_tasks_tucker.py
  --checkpoint-run-dir "${RUN_DIR}"
  --data-root "${DATA_ROOT}"
  --datasets "${DATASETS}"
  --tasks "${TASKS}"
  --shots "${SHOTS}"
)

if [[ -n "${GPUS}" ]]; then
  cmd+=(--gpus "${GPUS}")
fi

if [[ -n "${CHECKPOINT_NAME_PREFIX}" ]]; then
  cmd+=(--checkpoint-name-prefix "${CHECKPOINT_NAME_PREFIX}")
fi

cmd+=("$@")
cmd+=(
  --
  --emb_dim "${EMB_DIM}"
  --layers "${LAYERS}"
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
