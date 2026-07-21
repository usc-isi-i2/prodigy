#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/dataMeR1/phil/gfm/prodigy}"
DATA_ROOT="${DATA_ROOT:-/dataMeR1/phil/data}"
LOG_ROOT="${LOG_ROOT:-/dataMeR1/phil/logs}"
CONDA_SH="${CONDA_SH:-/home/mhchu/miniconda3/etc/profile.d/conda.sh}"
GPUS="${GPUS:-1}"
MODEL_NAME="${MODEL_NAME:-cp_hk_twitter_nm_bio_best}"
MODEL_PATH="${MODEL_PATH:-/dataMeR1/phil/gfm/prodigy/state/cp_hk_twitter_nm_bio_02_07_2026_08_58_57/state_dict}"
WORKERS="${WORKERS:-4}"
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

NM_DATASETS="${NM_DATASETS:-covid19_twitter,ukr_rus_twitter,midterm,cp_hk_twitter,twibot20,covid_political,election2020,ukr_rus_suspended}"
NC_DATASETS="${NC_DATASETS:-twibot20,covid_political,election2020,ukr_rus_suspended}"

mkdir -p "${LOG_ROOT}"

if [[ ! -s "${MODEL_PATH}" ]]; then
  echo "Missing model checkpoint: ${MODEL_PATH}" >&2
  exit 1
fi

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

model_list="$(mktemp "${LOG_ROOT}/cp_hk_transfer_model_list.XXXXXX.txt")"
printf "%s %s\n" "${MODEL_NAME}" "${MODEL_PATH}" > "${model_list}"

common_args=(
  --model-list "${model_list}"
  --data-root "${DATA_ROOT}"
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

echo "[run] CP-HK transfer NM, 3-way"
python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
  "${common_args[@]}" \
  --datasets "${NM_DATASETS}" \
  --tasks nm \
  --nm-n-way 3

echo "[run] CP-HK transfer NM, 30-way"
python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
  "${common_args[@]}" \
  --datasets "${NM_DATASETS}" \
  --tasks nm \
  --nm-n-way 30

echo "[run] CP-HK transfer NC"
python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
  "${common_args[@]}" \
  --datasets "${NC_DATASETS}" \
  --tasks nc
