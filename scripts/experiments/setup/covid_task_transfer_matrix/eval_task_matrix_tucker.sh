#!/usr/bin/env bash
# Evaluate the COVID task-transfer 3x3 matrix: train task x eval task.
# Requires model_list.txt from make_model_list.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MODEL_LIST="${MODEL_LIST:-${SCRIPT_DIR}/model_list.txt}"

DATA_ROOT="${DATA_ROOT:-/dataMeR1/phil/data}"
ROOT="${ROOT:-${DATA_ROOT}/covid19_twitter/graphs}"
GRAPH_FILENAME="${GRAPH_FILENAME:-retweet_graph_parquet.pt}"
PYTHON="${PYTHON:-python3}"
DEVICE="${DEVICE:-0}"
GPUS="${GPUS:-}"
DRY_RUN="${DRY_RUN:-0}"

WORKERS="${WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-0}"
DATASET_LEN_CAP="${DATASET_LEN_CAP:-1000}"
VAL_LEN_CAP="${VAL_LEN_CAP:-200}"
TEST_LEN_CAP="${TEST_LEN_CAP:-200}"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate prodigy
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
elif [[ "${DRY_RUN}" == "1" ]]; then
  echo "WARN: conda not found; continuing because DRY_RUN=1" >&2
else
  echo "ERROR: conda not found; run this on Tucker with the prodigy environment available" >&2
  exit 1
fi

cd "${REPO_ROOT}"

if [[ ! -s "${MODEL_LIST}" ]]; then
  echo "ERROR: missing model list: ${MODEL_LIST}" >&2
  echo "Run: STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ${SCRIPT_DIR}/make_model_list.sh" >&2
  exit 1
fi

common_args() {
  local model="$1"
  local ckpt="$2"
  local eval_task="$3"
  printf '%q ' \
    "${PYTHON}" experiments/run_single_experiment.py \
    --dataset covid19_twitter \
    --root "${ROOT}" \
    --graph_filename "${GRAPH_FILENAME}" \
    --edge_view temporal_history \
    --feature_subset all \
    --original_features True \
    --input_dim 768 \
    --workers "${WORKERS}" \
    --batch_size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --dataset_len_cap "${DATASET_LEN_CAP}" \
    --val_len_cap "${VAL_LEN_CAP}" \
    --test_len_cap "${TEST_LEN_CAP}" \
    --epochs 1 \
    --eval_step 1 \
    --checkpoint_step 1 \
    --eval_only True \
    --eval_test_before_train True \
    --pretrained_model_run "${ckpt}" \
    --prefix "eval_covid_task_transfer_${model}_to_${eval_task}"
}

task_args() {
  local eval_task="$1"
  case "${eval_task}" in
    nm)
      printf '%q ' --task_name nm --n_way "${NM_N_WAY:-3}" --n_shots "${NM_N_SHOTS:-3}" --n_query "${NM_N_QUERY:-8}" --zero_shot False
      ;;
    cl)
      printf '%q ' --task_name cl --augmentation "${CL_AUGMENTATION:-NZ0.2}" --augment_test True --n_way "${CL_N_WAY:-16}" --n_shots "${CL_N_SHOTS:-1}" --n_query "${CL_N_QUERY:-1}" --zero_shot False
      ;;
    fp)
      printf '%q ' --task_name fp --fp_mask_ratio "${FP_MASK_RATIO:-0.3}" --fp_mask_strategy "${FP_MASK_STRATEGY:-zero}" --n_way "${FP_N_WAY:-8}" --n_shots "${FP_N_SHOTS:-1}" --n_query "${FP_N_QUERY:-1}" --zero_shot False
      ;;
    *)
      echo "ERROR: unknown eval task: ${eval_task}" >&2
      exit 2
      ;;
  esac
}

eval_tasks=(nm cl fp)
jobs=()

while IFS= read -r row || [[ -n "${row}" ]]; do
  [[ -n "${row}" && "${row}" != \#* ]] || continue
  model="$(awk '{print $1}' <<< "${row}")"
  ckpt="$(awk '{print $2}' <<< "${row}")"
  for eval_task in "${eval_tasks[@]}"; do
    jobs+=("$(common_args "${model}" "${ckpt}" "${eval_task}") $(task_args "${eval_task}")")
  done
done < "${MODEL_LIST}"

IFS=',' read -r -a gpu_slots <<< "${GPUS}"
if [[ -z "${GPUS}" ]]; then
  gpu_slots=("")
fi

echo "[progress] jobs=${#jobs[@]} parallel_slots=${#gpu_slots[@]} gpus=${GPUS:-<inherit>}"

running_pids=()
running_labels=()
failures=0

wait_one() {
  local pid="${running_pids[0]}"
  local label="${running_labels[0]}"
  if ! wait "${pid}"; then
    echo "[error] ${label}" >&2
    failures=$((failures + 1))
  else
    echo "[done] ${label}"
  fi
  running_pids=("${running_pids[@]:1}")
  running_labels=("${running_labels[@]:1}")
}

for i in "${!jobs[@]}"; do
  slot_idx=$((i % ${#gpu_slots[@]}))
  gpu="${gpu_slots[${slot_idx}]}"
  label="job $((i + 1))/${#jobs[@]}"
  cmd="${jobs[$i]}"

  if [[ -n "${gpu}" ]]; then
    cmd="CUDA_VISIBLE_DEVICES=${gpu} ${cmd}"
    label="${label} gpu=${gpu}"
  fi

  echo "[cmd] ${cmd}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi

  eval "${cmd}" &
  running_pids+=("$!")
  running_labels+=("${label}")

  if [[ "${#running_pids[@]}" -ge "${#gpu_slots[@]}" ]]; then
    wait_one
  fi
done

while [[ "${#running_pids[@]}" -gt 0 ]]; do
  wait_one
done

exit "${failures}"
