#!/usr/bin/env bash
# Two GPU workers: train one final-core rung, evaluate it, then take the next rung.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state/nm_ladder_global_finalcore}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/nm_ladder_global_finalcore}"
RUN_STAMP="${RUN_STAMP:-20260814global}"
GPUS_TEXT="${GPUS:-0 1}"
read -r -a GPU_IDS <<< "${GPUS_TEXT}"
[[ "${GPU_IDS[*]}" == "0 1" ]] || { echo 'GPUS must be exactly "0 1"' >&2; exit 2; }

export PATH="/home/mhchu/miniconda3/bin:${PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
PYTHON="${CONDA_PREFIX}/bin/python"
mkdir -p "${STATE_ROOT}" "${LOG_ROOT}/train" "${LOG_ROOT}/eval" "${LOG_ROOT}/launch"
cd "${REPO_ROOT}"
"${PYTHON}" "${SCRIPT_DIR}/make_configs.py" --check >/dev/null

jobs=()
while IFS=$'\t' read -r rung model_id job_index _newcomer sources config; do
  [[ "${rung}" == rung ]] && continue
  jobs+=("${rung}:${model_id}:${job_index}:${sources}:${config}")
done < "${SCRIPT_DIR}/manifest.tsv"
[[ ${#jobs[@]} -eq 8 ]] || { echo "expected eight rung jobs" >&2; exit 2; }

if [[ "${DRY_RUN:-0}" != 1 ]]; then
  for gpu in 0 1; do
    used="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    (( used < 1000 )) || { echo "GPU ${gpu} busy (${used} MiB)" >&2; exit 1; }
  done
  available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  (( available_kib >= 350 * 1024 * 1024 )) || {
    echo "need at least 350 GiB available host RAM for two graph loads" >&2; exit 1;
  }
fi

worker() {
  local worker_index="$1" gpu="$2" index=0 item rung model_id job_index sources config
  for item in "${jobs[@]}"; do
    if (( index % 2 == worker_index )); then
      IFS=: read -r rung model_id job_index sources config <<< "${item}"
      prefix="finalcore_${model_id}_s0"
      run_name="${prefix}_${RUN_STAMP}"
      checkpoint="${STATE_ROOT}/${run_name}/checkpoint/state_dict_2500.ckpt"
      train_log="${LOG_ROOT}/train/${run_name}.log"
      eval_log="${LOG_ROOT}/eval/${run_name}.log"
      if [[ ! -f "${checkpoint}" ]]; then
        [[ ! -e "${STATE_ROOT}/${run_name}" ]] || {
          echo "[gpu ${gpu}] REFUSE incomplete state ${STATE_ROOT}/${run_name}" >&2; return 1;
        }
        cmd=("${PYTHON}" -u experiments/run_single_experiment.py
          --config "${SCRIPT_DIR}/${config}" --device "${gpu}"
          --prefix "${prefix}" --timestamp "${RUN_STAMP}"
          --state_dir "${STATE_ROOT}" --log_dir "${LOG_ROOT}")
        if [[ "${DRY_RUN:-0}" == 1 ]]; then
          printf 'DRY TRAIN rung=%s gpu=%s' "${rung}" "${gpu}"; printf ' %q' "${cmd[@]}"; printf '\n'
        else
          echo "[gpu ${gpu}] TRAIN_START rung=${rung} model=${model_id} utc=$(date -u +%FT%TZ)"
          "${cmd[@]}" > "${train_log}" 2>&1
          [[ -f "${checkpoint}" ]] || { echo "missing ${checkpoint}" >&2; return 1; }
          echo "[gpu ${gpu}] TRAIN_DONE rung=${rung} utc=$(date -u +%FT%TZ)"
        fi
      else
        echo "[gpu ${gpu}] TRAIN_SKIP complete rung=${rung}"
      fi
      if [[ "${DRY_RUN:-0}" == 1 ]]; then
        echo "DRY EVAL rung=${rung} gpu=${gpu} job_index=${job_index}"
      else
        echo "[gpu ${gpu}] EVAL_START rung=${rung} utc=$(date -u +%FT%TZ)"
        STATE_ROOT="${STATE_ROOT}" LOG_ROOT="${LOG_ROOT}" RUN_STAMP="${RUN_STAMP}" \
          "${SCRIPT_DIR}/eval_checkpoint_tucker.sh" "${job_index}" "${gpu}" \
          > "${eval_log}" 2>&1
        echo "[gpu ${gpu}] EVAL_DONE rung=${rung} utc=$(date -u +%FT%TZ)"
      fi
    fi
    ((index+=1))
  done
}

{
  echo "protocol=finalcore_global_orderA_seed0_train_then_fixed_test_v1"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "gpus=0,1"
  echo "run_stamp=${RUN_STAMP}"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "${LOG_ROOT}/launch/provenance.txt"

pids=()
worker 0 0 & pids+=("$!")
worker 1 1 & pids+=("$!")
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"
