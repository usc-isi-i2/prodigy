#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
mkdir -p "${SCRIPT_DIR}/run_logs"
read -r -a GPU_ARR <<< "${GPUS:-0 1}"
for gpu in "${GPU_ARR[@]}"; do
  [[ "${gpu}" =~ ^[01]$ ]] || { echo "only Tucker GPUs 0 and 1 are owned" >&2; exit 2; }
done
export PATH="/home/mhchu/miniconda3/bin:${PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"
python3 "${SCRIPT_DIR}/make_plan.py" --check
stamp="$(date +%Y%m%d_%H%M%S)"
model_timeout="${MODEL_TIMEOUT_SECONDS:-1800}"
model_limit="${MODEL_LIMIT:-0}"
jobs=()
while IFS=$'\t' read -r _size _codes _donors _heldout prefix; do
  [[ "${prefix}" == "prefix" ]] && continue
  jobs+=("${prefix}")
done < "${SCRIPT_DIR}/manifest.tsv"

worker() {
  local index="$1" gpu="$2" prefix job_index=0 selected=0
  local log="${SCRIPT_DIR}/run_logs/train_shard${index}_gpu${gpu}_${stamp}.log"
  for prefix in "${jobs[@]}"; do
    if (( job_index % ${#GPU_ARR[@]} == index )); then
      if (( model_limit > 0 && selected >= model_limit )); then
        break
      fi
      selected=$((selected + 1))
      cmd=(python3 -u "${SCRIPT_DIR}/run_train.py" --device "${gpu}"
           --model-prefix "${prefix}")
      [[ "${DRY_RUN:-0}" != 1 ]] || cmd+=(--dry-run)
      echo "[gpu ${gpu}] START ${prefix} $(date -u +%FT%TZ)" | tee -a "${log}"
      if [[ "${DRY_RUN:-0}" == 1 ]]; then
        "${cmd[@]}" | tee -a "${log}"
      else
        timeout --signal=TERM --kill-after=60 "${model_timeout}" \
          "${cmd[@]}" >>"${log}" 2>&1
      fi
      echo "[gpu ${gpu}] DONE ${prefix} $(date -u +%FT%TZ)" | tee -a "${log}"
    fi
    job_index=$((job_index + 1))
  done
}

pids=()
for index in "${!GPU_ARR[@]}"; do
  worker "${index}" "${GPU_ARR[$index]}" & pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done
exit "${rc}"
