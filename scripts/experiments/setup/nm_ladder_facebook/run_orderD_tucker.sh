#!/usr/bin/env bash
# Train the three new Order D rungs concurrently, then evaluate them on all nine graphs.
# Rungs 1-5 reuse Order A and rung 9 reuses the existing all-nine model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/dataMeR1/phil/data}"
TRAIN_GPUS="${TRAIN_GPUS:-0 2 3}"
EVAL_GPUS="${EVAL_GPUS:-0,2,3}"
STEP="${STEP:-40000}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${SCRIPT_DIR}/run_logs/orderD_${STAMP}"
STATUS="${RUN_DIR}/status.txt"
MODEL_LIST="${RUN_DIR}/orderD_models.txt"
DATASETS="ukr_rus_twitter,covid19_twitter,midterm,covid_political,election2020,ukr_rus_suspended,twibot20,cp_hk_twitter,facebook_page_reference"
FB_GRAPH_OVERRIDE="facebook_page_reference=page_reference_structural.pt"
CONFIGS=(train_ordD_r6.yaml train_ordD_r7.yaml train_ordD_r8.yaml)
PREFIXES=(nm_ladder_ordD_r6_facebook nm_ladder_ordD_r7_facebook nm_ladder_ordD_r8_facebook)
LABELS=(nm_ladder_ordD_r6 nm_ladder_ordD_r7 nm_ladder_ordD_r8)

mkdir -p "${RUN_DIR}"

fail() {
  printf 'FAILED: %s\n' "$*" | tee -a "${STATUS}" >&2
  exit 1
}

latest_checkpoint() {
  local prefix="$1"
  local state_dir=""
  local checkpoint=""
  state_dir="$(ls -dt "${REPO_ROOT}/state/${prefix}_"*/ 2>/dev/null | head -n 1 || true)"
  [[ -n "${state_dir}" ]] || return 1
  checkpoint="${state_dir}checkpoint/state_dict_${STEP}.ckpt"
  [[ -f "${checkpoint}" ]] || return 1
  printf '%s\n' "${checkpoint}"
}

check_gpu_idle() {
  local gpu="$1"
  local used=""
  used="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits)"
  [[ "${used}" =~ ^[0-9]+$ ]] || fail "could not read GPU ${gpu} memory"
  (( used < 1024 )) || fail "GPU ${gpu} is not idle (${used} MiB used)"
}

read -r -a GPU_ARR <<< "${TRAIN_GPUS}"
[[ "${#GPU_ARR[@]}" == "${#CONFIGS[@]}" ]] \
  || fail "need exactly three training GPUs; got: ${TRAIN_GPUS}"

[[ -f "${DATA_ROOT}/merged/graphs/ukr_rus_covid_midterm_all9_facebook_graph.pt" ]] \
  || fail "missing all-nine merged graph"
[[ -f "${DATA_ROOT}/facebook_page_reference/graphs/page_reference_structural.pt" ]] \
  || fail "missing structural Facebook graph"

for i in "${!CONFIGS[@]}"; do
  [[ -f "${SCRIPT_DIR}/${CONFIGS[$i]}" ]] || fail "missing ${CONFIGS[$i]}"
  check_gpu_idle "${GPU_ARR[$i]}"
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY RUN OK\n'
  for i in "${!CONFIGS[@]}"; do
    printf 'train %s on GPU %s\n' "${CONFIGS[$i]}" "${GPU_ARR[$i]}"
  done
  printf 'then evaluate 3 models x 9 graphs on GPUs %s\n' "${EVAL_GPUS}"
  exit 0
fi

printf 'RUNNING phase 1: Order D rungs 6-8 training in parallel\n' | tee "${STATUS}"
declare -a TRAIN_PIDS
for i in "${!CONFIGS[@]}"; do
  config="${SCRIPT_DIR}/${CONFIGS[$i]}"
  gpu="${GPU_ARR[$i]}"
  (
    export PATH="/home/mhchu/miniconda3/bin:${PATH}"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate prodigy
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    cd "${REPO_ROOT}"
    python3 experiments/run_single_experiment.py --config "${config}" --device "${gpu}"
  ) > "${RUN_DIR}/train_${LABELS[$i]}_gpu${gpu}.log" 2>&1 &
  TRAIN_PIDS+=("$!")
done

train_rc=0
for i in "${!TRAIN_PIDS[@]}"; do
  if ! wait "${TRAIN_PIDS[$i]}"; then
    printf 'training failed: %s\n' "${CONFIGS[$i]}" | tee -a "${STATUS}" >&2
    train_rc=1
  fi
done
(( train_rc == 0 )) || fail "one or more Order D training runs failed"

: > "${MODEL_LIST}"
for i in "${!PREFIXES[@]}"; do
  checkpoint="$(latest_checkpoint "${PREFIXES[$i]}")" \
    || fail "missing ${PREFIXES[$i]} state_dict_${STEP}.ckpt"
  printf '%s %s\n' "${LABELS[$i]}" "${checkpoint}" >> "${MODEL_LIST}"
done

printf 'RUNNING phase 2: Order D rungs 6-8 to all nine eval graphs\n' | tee -a "${STATUS}"
(
  export PATH="/home/mhchu/miniconda3/bin:${PATH}"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate prodigy
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  cd "${REPO_ROOT}"
  python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
    --model-list "${MODEL_LIST}" \
    --data-root "${DATA_ROOT}" \
    --datasets "${DATASETS}" \
    --graph-filenames "${FB_GRAPH_OVERRIDE}" \
    --tasks nm \
    --shots 3 \
    --nm-n-way 30 \
    --gpus "${EVAL_GPUS}"
) > "${RUN_DIR}/eval_orderD_to_all9.log" 2>&1 \
  || fail "Order D all-nine evaluation failed"

printf 'COMPLETE models=%s\n' "${MODEL_LIST}" | tee -a "${STATUS}"
printf 'Artifacts and logs: %s\n' "${RUN_DIR}" | tee -a "${STATUS}"
