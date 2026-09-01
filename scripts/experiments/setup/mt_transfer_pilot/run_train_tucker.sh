#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
MODE="${MODE:-full}"
SLOTS="${SLOTS:-2,3,2,3,2,3}"
IFS=',' read -r -a SLOT_ARR <<< "${SLOTS}"
for gpu in "${SLOT_ARR[@]}"; do
  [[ "${gpu}" == 2 || "${gpu}" == 3 ]] || { echo "only GPUs 2 and 3 are allowed" >&2; exit 2; }
done

SOURCES=(
  'covid_political|/dataMeR1/phil/data/covid_political/graphs|retweet_graph.pt'
  'election2020|/dataMeR1/phil/data/election2020/graphs|retweet_graph.pt'
  'facebook_page_reference|/dataMeR1/phil/data/facebook_page_reference/graphs|page_reference_graph.pt'
  'twibot20|/dataMeR1/phil/data/twibot20/graphs|retweet_graph.pt'
  'ukr_rus_suspended|/dataMeR1/phil/data/ukr_rus_suspended/graphs|retweet_graph.pt'
)
IFS=',' read -r -a ARMS <<< "${ARMS_CSV:-MT,NM_MT}"
for arm in "${ARMS[@]}"; do
  [[ "${arm}" == MT || "${arm}" == NM || "${arm}" == NM_MT ]] || { echo "unknown arm: ${arm}" >&2; exit 2; }
done
JOBS=()
for arm in "${ARMS[@]}"; do
  for spec in "${SOURCES[@]}"; do JOBS+=("${arm}|${spec}"); done
done

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
cd "${REPO_ROOT}"
mkdir -p "${SCRIPT_DIR}/run_logs"

run_one() {
  local job="$1" gpu="$2" arm dataset root filename
  IFS='|' read -r arm dataset root filename <<< "${job}"
  local extra=()
  if [[ "${MODE}" == smoke ]]; then
    extra=(--dataset_len_cap 12 --val_len_cap 4 --test_len_cap 4 --checkpoint_step 12 --prefix "mtpilot_${arm}_${dataset}_smoke")
  else
    extra=(--prefix "mtpilot_${arm}_${dataset}")
  fi
  local log="${SCRIPT_DIR}/run_logs/${MODE}_${arm}_${dataset}_gpu${gpu}_$(date +%Y%m%d_%H%M%S).log"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 experiments/run_single_experiment.py \
    --config "${SCRIPT_DIR}/configs/${arm}.yaml" --dataset "${dataset}" \
    --root "${root}" --graph_filename "${filename}" --device 0 "${extra[@]}" >"${log}" 2>&1
}

if [[ "${MODE}" == smoke ]]; then
  JOBS=()
  for arm in "${ARMS[@]}"; do JOBS+=("${arm}|${SOURCES[0]}"); done
fi
next=0
while (( next < ${#JOBS[@]} )); do
  pids=(); labels=()
  for slot in "${!SLOT_ARR[@]}"; do
    (( next >= ${#JOBS[@]} )) && break
    run_one "${JOBS[$next]}" "${SLOT_ARR[$slot]}" &
    pids+=("$!"); labels+=("${JOBS[$next]}"); ((next+=1))
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "FAILED ${labels[$i]}" >&2; exit 1; }
    echo "DONE ${labels[$i]}" >&2
  done
done
