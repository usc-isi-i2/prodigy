#!/usr/bin/env bash
# Expand the historical 8x8 one-hop NM specialist matrix with Facebook.
#
# Phase 1 runs concurrently:
#   GPU 0   : train the Facebook-only specialist to the matched 40k checkpoint
#   GPUs 1-3: evaluate the eight existing specialists on Facebook
# Phase 2 uses GPUs 0-3 to evaluate the Facebook specialist on all nine graphs.
#
# Run from a detached tmux session. Runtime lists and logs are written beneath
# this setup folder's gitignored run_logs/ directory.
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

CONFIG="${SCRIPT_DIR}/facebook_page_reference.yaml"
EXISTING_MODELS="${RUN_DIR}/existing_specialists.txt"
FACEBOOK_MODEL="${RUN_DIR}/facebook_specialist.txt"
STATUS="${RUN_DIR}/status.txt"

PREFIXES=(
  nm_ss_ukr_rus_twitter
  nm_ss_covid19_twitter
  nm_ss_midterm
  nm_ss_covid_political
  nm_ss_election2020
  nm_ss_ukr_rus_suspended
  nm_ss_twibot20
  nm_ss_cp_hk_twitter
)

DATASETS="ukr_rus_twitter,covid19_twitter,midterm,covid_political,election2020,ukr_rus_suspended,twibot20,cp_hk_twitter,facebook_page_reference"
FB_GRAPH_OVERRIDE="facebook_page_reference=page_reference_structural.pt"

fail() {
  printf 'FAILED: %s\n' "$*" | tee -a "${STATUS}" >&2
  exit 1
}

latest_checkpoint() {
  local state_root="$1"
  local prefix="$2"
  local checkpoint=""
  local run_dir=""
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
[[ -f "${DATA_ROOT}/facebook_page_reference/graphs/page_reference_structural.pt" ]] \
  || fail "missing structural Facebook graph"

for gpu in 0 1 2 3; do
  check_gpu_idle "${gpu}"
done

: > "${EXISTING_MODELS}"
for prefix in "${PREFIXES[@]}"; do
  checkpoint="$(latest_checkpoint "${MAIN_STATE_DIR}" "${prefix}")" \
    || fail "missing ${prefix} state_dict_${STEP}.ckpt under ${MAIN_STATE_DIR}"
  printf '%s %s\n' "${prefix}" "${checkpoint}" >> "${EXISTING_MODELS}"
done
[[ "$(wc -l < "${EXISTING_MODELS}" | tr -d ' ')" == "8" ]] \
  || fail "expected eight existing specialists"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY RUN OK\n'
  printf 'training: Facebook specialist on GPU %s\n' "${TRAIN_GPU}"
  printf 'column: 8 specialists -> Facebook on GPUs %s\n' "${COLUMN_GPUS}"
  printf 'row: Facebook specialist -> 9 targets on GPUs %s\n' "${ROW_GPUS}"
  printf 'existing model list:\n'
  cat "${EXISTING_MODELS}"
  exit 0
fi

printf 'RUNNING phase 1: Facebook train + existing-to-Facebook eval\n' | tee "${STATUS}"

(
  export PATH="/home/mhchu/miniconda3/bin:${PATH}"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate prodigy
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  cd "${REPO_ROOT}"
  python3 experiments/run_single_experiment.py \
    --config "${CONFIG}" \
    --device "${TRAIN_GPU}"
) > "${RUN_DIR}/train_facebook.log" 2>&1 &
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
    --gpus "${COLUMN_GPUS}" \
    --continue-on-error
) > "${RUN_DIR}/eval_existing_to_facebook.log" 2>&1 &
COLUMN_PID=$!

train_rc=0
column_rc=0
wait "${TRAIN_PID}" || train_rc=$?
wait "${COLUMN_PID}" || column_rc=$?
(( train_rc == 0 )) || fail "Facebook training exited ${train_rc}"
(( column_rc == 0 )) || fail "existing-to-Facebook evaluation exited ${column_rc}"

facebook_checkpoint="$(latest_checkpoint "${REPO_ROOT}/state" nm_ss_facebook_page_reference)" \
  || fail "Facebook training completed without state_dict_${STEP}.ckpt"
printf 'nm_ss_facebook_page_reference %s\n' "${facebook_checkpoint}" > "${FACEBOOK_MODEL}"

printf 'RUNNING phase 2: Facebook-to-all-nine eval\n' | tee -a "${STATUS}"
(
  export PATH="/home/mhchu/miniconda3/bin:${PATH}"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate prodigy
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  cd "${REPO_ROOT}"
  python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
    --model-list "${FACEBOOK_MODEL}" \
    --data-root "${DATA_ROOT}" \
    --datasets "${DATASETS}" \
    --graph-filenames "${FB_GRAPH_OVERRIDE}" \
    --tasks nm \
    --shots 3 \
    --nm-n-way 30 \
    --gpus "${ROW_GPUS}" \
    --continue-on-error
) > "${RUN_DIR}/eval_facebook_to_all.log" 2>&1 \
  || fail "Facebook-to-all-nine evaluation failed"

printf 'COMPLETE checkpoint=%s\n' "${facebook_checkpoint}" | tee -a "${STATUS}"
printf 'Artifacts and logs: %s\n' "${RUN_DIR}" | tee -a "${STATUS}"
