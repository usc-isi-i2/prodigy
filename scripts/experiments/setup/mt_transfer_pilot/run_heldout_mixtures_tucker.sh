#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
cd "${ROOT}"

SLOTS_CSV="${SLOTS:-2,3,2,3}"
IFS=',' read -r -a SLOTS_ARR <<< "${SLOTS_CSV}"
TARGETS=(covid_political election2020 facebook_page_reference twibot20 ukr_rus_suspended)
ARMS=(MT NM NM_MT)
JOBS=()
for arm in "${ARMS[@]}"; do
  for target in "${TARGETS[@]}"; do JOBS+=("${arm}|${target}"); done
done
if [[ "${SMOKE:-0}" == 1 ]]; then JOBS=("MT|covid_political" "NM|covid_political" "NM_MT|covid_political"); fi

run_one() {
  local job="$1" gpu="$2" arm target
  IFS='|' read -r arm target <<< "${job}"
  local extra=()
  [[ "${SMOKE:-0}" == 1 ]] && extra+=(--smoke)
  CUDA_VISIBLE_DEVICES="${gpu}" python3 "${SCRIPT_DIR}/run_heldout_mixture.py" \
    --arm "${arm}" --heldout "${target}" --device 0 "${extra[@]}" \
    > "${SCRIPT_DIR}/run_logs/heldout_${arm}_${target}_gpu${gpu}_$(date +%Y%m%d_%H%M%S).log" 2>&1
}

next=0
while (( next < ${#JOBS[@]} )); do
  pids=(); labels=()
  for slot in "${!SLOTS_ARR[@]}"; do
    (( next >= ${#JOBS[@]} )) && break
    run_one "${JOBS[$next]}" "${SLOTS_ARR[$slot]}" &
    pids+=("$!"); labels+=("${JOBS[$next]}"); ((next+=1))
  done
  for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "FAILED ${labels[$i]}" >&2; exit 1; }
    echo "DONE ${labels[$i]}" >&2
  done
done
