#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
cd "${ROOT}"

mkdir -p "${SCRIPT_DIR}/run_logs"
IFS=',' read -r -a SLOTS_ARR <<< "${SLOTS:-2,3,2,3}"
TARGETS=(covid_political election2020 facebook_page_reference twibot20 ukr_rus_suspended)
[[ "${SMOKE:-0}" == 1 ]] && TARGETS=(twibot20)

run_one() {
  local target="$1" gpu="$2" extra=()
  [[ "${SMOKE:-0}" == 1 ]] && extra+=(--smoke)
  CUDA_VISIBLE_DEVICES="${gpu}" python3 "${SCRIPT_DIR}/run_heldout_mixture.py" \
    --arm NM_MT --heldout "${target}" --device 0 \
    --task-embedding-dim 8 --task-embedding-dropout 0.25 \
    --task-embedding-fusion film "${extra[@]}" \
    > "${SCRIPT_DIR}/run_logs/film8_${target}_gpu${gpu}_$(date +%Y%m%d_%H%M%S).log" 2>&1
}

next=0
while (( next < ${#TARGETS[@]} )); do
  pids=(); labels=()
  for slot in "${!SLOTS_ARR[@]}"; do
    (( next >= ${#TARGETS[@]} )) && break
    run_one "${TARGETS[$next]}" "${SLOTS_ARR[$slot]}" &
    pids+=("$!"); labels+=("${TARGETS[$next]}"); ((next+=1))
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "FAILED ${labels[$i]}" >&2; exit 1; }
    echo "DONE ${labels[$i]}" >&2
  done
done
