#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/dataMeR1/phil/gfm/prodigy}"
DATA_ROOT="${DATA_ROOT:-/dataMeR1/phil/data}"
LOG_ROOT="${LOG_ROOT:-/dataMeR1/phil/logs}"
CONDA_SH="${CONDA_SH:-/home/mhchu/miniconda3/etc/profile.d/conda.sh}"
GPUS="${GPUS:-1}"
WORKERS="${WORKERS:-4}"
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
MODEL_LIST="${MODEL_LIST:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${LOG_ROOT}"

IFS=',' read -r -a gpu_list <<< "${GPUS}"
for gpu in "${gpu_list[@]}"; do
  case "${gpu}" in
    0|1|2|3) ;;
    *)
      echo "Refusing GPU '${gpu}'. Tucker evals may only use physical GPUs 0,1,2,3." >&2
      exit 1
      ;;
  esac
done

source "${CONDA_SH}"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"

if [[ -z "${MODEL_LIST}" ]]; then
  MODEL_LIST="${LOG_ROOT}/cp_hk_transfer_in_model_list_matched.txt"
  OUT="${MODEL_LIST}" STATE_DIR="${REPO_ROOT}/state" "${SCRIPT_DIR}/make_model_list_matched.sh"
fi

common_args=(
  --model-list "${MODEL_LIST}"
  --data-root "${DATA_ROOT}"
  --datasets cp_hk_twitter
  --tasks nm
  --shots 3
  --gpus "${GPUS}"
  --workers "${WORKERS}"
  --device 0
)

if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
  common_args+=(--continue-on-error)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  common_args+=(--dry-run)
fi

echo "[run] source NM models -> CP-HK, 3-way"
python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
  "${common_args[@]}" \
  --nm-n-way 3

echo "[run] source NM models -> CP-HK, 30-way"
python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
  "${common_args[@]}" \
  --nm-n-way 30
