#!/usr/bin/env bash
# Append Facebook as Order A rung 9 while backfilling the Facebook eval column.
#
# Phase 1 runs concurrently:
#   GPU 0   : train rung 9 on the all-nine merge
#   GPUs 1-3: evaluate historical Order A rungs 1-8 on Facebook
# Phase 2 uses GPUs 0-3 to evaluate rung 9 on all nine individual graphs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
MAIN_STATE_DIR="${MAIN_STATE_DIR:-/dataMeR1/phil/gfm/prodigy/state}"
DATA_ROOT="${DATA_ROOT:-/dataMeR1/phil/data}"
TRAIN_GPU="${TRAIN_GPU:-0}"
COLUMN_GPUS="${COLUMN_GPUS:-1,2,3}"
ROW_GPUS="${ROW_GPUS:-0,1,2,3}"
STEP="${STEP:-40000}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${SCRIPT_DIR}/run_logs/${STAMP}"
mkdir -p "${RUN_DIR}"

CONFIG="${SCRIPT_DIR}/train_ordA_r9.yaml"
ALL9_GRAPH="${DATA_ROOT}/merged/graphs/ukr_rus_covid_midterm_all9_facebook_graph.pt"
EXISTING_MODELS="${RUN_DIR}/existing_orderA_rungs.txt"
RUNG9_MODEL="${RUN_DIR}/rung9_model.txt"
STATUS="${RUN_DIR}/status.txt"
DATASETS="ukr_rus_twitter,covid19_twitter,midterm,covid_political,election2020,ukr_rus_suspended,twibot20,cp_hk_twitter,facebook_page_reference"
FB_GRAPH_OVERRIDE="facebook_page_reference=page_reference_structural.pt"

# label|historical run prefix
RUNG_SPECS=(
  "nm_ladder_ordA_r1|ukr_only_nm"
  "nm_ladder_ordA_r2|merged_ukr_rus_covid_nm_wb"
  "nm_ladder_ordA_r3|merged_ukr_rus_covid_midterm_nm_wb"
  "nm_ladder_ordA_r4|nm_ladder_4src"
  "nm_ladder_ordA_r5|nm_ladder_5src"
  "nm_ladder_ordA_r6|nm_ladder_6src"
  "nm_ladder_ordA_r7|nm_ladder_7src"
  "nm_ladder_ordA_r8|merged_ukr_rus_covid_midterm_all8_nm_wb"
)

fail() {
  printf 'FAILED: %s\n' "$*" | tee -a "${STATUS}" >&2
  exit 1
}

latest_checkpoint() {
  local state_root="$1"
  local prefix="$2"
  local run_dir=""
  local checkpoint=""
  run_dir="$(ls -dt "${state_root}/${prefix}_"*/ 2>/dev/null | head -n 1 || true)"
  [[ -n "${run_dir}" ]] || return 1
  checkpoint="${run_dir}checkpoint/state_dict_${STEP}.ckpt"
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

[[ -f "${CONFIG}" ]] || fail "missing config ${CONFIG}"
[[ -f "${ALL9_GRAPH}" ]] || fail "missing all-nine graph ${ALL9_GRAPH}"
[[ -f "${DATA_ROOT}/facebook_page_reference/graphs/page_reference_structural.pt" ]] \
  || fail "missing structural Facebook graph"

for gpu in 0 1 2 3; do
  check_gpu_idle "${gpu}"
done

: > "${EXISTING_MODELS}"
for spec in "${RUNG_SPECS[@]}"; do
  label="${spec%%|*}"
  prefix="${spec#*|}"
  checkpoint="$(latest_checkpoint "${MAIN_STATE_DIR}" "${prefix}")" \
    || fail "missing ${prefix} state_dict_${STEP}.ckpt under ${MAIN_STATE_DIR}"
  printf '%s %s\n' "${label}" "${checkpoint}" >> "${EXISTING_MODELS}"
done
[[ "$(wc -l < "${EXISTING_MODELS}" | tr -d ' ')" == "8" ]] \
  || fail "expected eight historical ladder rungs"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY RUN OK\n'
  printf 'training: Order A rung 9 on GPU %s\n' "${TRAIN_GPU}"
  printf 'column: rungs 1-8 -> Facebook on GPUs %s\n' "${COLUMN_GPUS}"
  printf 'row: rung 9 -> all nine on GPUs %s\n' "${ROW_GPUS}"
  printf 'historical rung model list:\n'
  cat "${EXISTING_MODELS}"
  exit 0
fi

printf 'RUNNING phase 1: rung-9 train + rungs-1-8-to-Facebook eval\n' | tee "${STATUS}"

(
  export PATH="/home/mhchu/miniconda3/bin:${PATH}"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate prodigy
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  cd "${REPO_ROOT}"
  python3 experiments/run_single_experiment.py \
    --config "${CONFIG}" \
    --device "${TRAIN_GPU}"
) > "${RUN_DIR}/train_rung9.log" 2>&1 &
TRAIN_PID=$!

(
  export PATH="/home/mhchu/miniconda3/bin:${PATH}"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate prodigy
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  cd "${REPO_ROOT}"
  python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
    --model-list "${EXISTING_MODELS}" \
    --data-root "${DATA_ROOT}" \
    --datasets facebook_page_reference \
    --graph-filenames "${FB_GRAPH_OVERRIDE}" \
    --tasks nm \
    --shots 3 \
    --nm-n-way 30 \
    --gpus "${COLUMN_GPUS}"
) > "${RUN_DIR}/eval_rungs_to_facebook.log" 2>&1 &
COLUMN_PID=$!

train_rc=0
column_rc=0
wait "${TRAIN_PID}" || train_rc=$?
wait "${COLUMN_PID}" || column_rc=$?
(( train_rc == 0 )) || fail "rung-9 training exited ${train_rc}"
(( column_rc == 0 )) || fail "rungs-to-Facebook evaluation exited ${column_rc}"

rung9_checkpoint="$(latest_checkpoint "${REPO_ROOT}/state" nm_ladder_ordA_r9_facebook)" \
  || fail "rung-9 training completed without state_dict_${STEP}.ckpt"
printf 'nm_ladder_ordA_r9_facebook %s\n' "${rung9_checkpoint}" > "${RUNG9_MODEL}"

printf 'RUNNING phase 2: rung-9-to-all-nine eval\n' | tee -a "${STATUS}"
(
  export PATH="/home/mhchu/miniconda3/bin:${PATH}"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate prodigy
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  cd "${REPO_ROOT}"
  python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
    --model-list "${RUNG9_MODEL}" \
    --data-root "${DATA_ROOT}" \
    --datasets "${DATASETS}" \
    --graph-filenames "${FB_GRAPH_OVERRIDE}" \
    --tasks nm \
    --shots 3 \
    --nm-n-way 30 \
    --gpus "${ROW_GPUS}"
) > "${RUN_DIR}/eval_rung9_to_all.log" 2>&1 \
  || fail "rung-9-to-all-nine evaluation failed"

printf 'COMPLETE checkpoint=%s\n' "${rung9_checkpoint}" | tee -a "${STATUS}"
printf 'Artifacts and logs: %s\n' "${RUN_DIR}" | tee -a "${STATUS}"
