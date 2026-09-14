#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/mhchu/miniconda3/envs/prodigy/bin/python3}"
CONFIG="${ROOT}/configs/twibot_strict_pilot.yaml"
SPLIT_ROOT="${ROOT}/state/twibot_strict_pilot/splits"
STATE_ROOT="${ROOT}/state/twibot_saturation"
RESULT_ROOT="${ROOT}/results/twibot_saturation"
LOG_ROOT="${ROOT}/log/twibot_saturation"
mkdir -p "${STATE_ROOT}" "${RESULT_ROOT}/raw" "${LOG_ROOT}"

declare -A MIX_SOURCES
MIX_SOURCES[mixture_k3]="covid_political,ukr_rus_suspended,election2020"
MIX_SOURCES[mixture_k6]="covid_political,ukr_rus_suspended,election2020,facebook_page_reference,cora,pubmed"
TRAIN_JOBS=()
for seed in 0 1 2; do
  for arm in mixture_k3 mixture_k6; do TRAIN_JOBS+=("${arm}|${seed}"); done
done

train_worker() {
  local index="$1" gpu="$2" job arm seed run_id cursor=0
  for job in "${TRAIN_JOBS[@]}"; do
    if (( cursor % 6 == index )); then
      IFS='|' read -r arm seed <<<"${job}"
      run_id="${arm}_existing_s${seed}"
      if [[ ! -f "${STATE_ROOT}/${run_id}/summary.json" ]]; then
        echo "[gpu ${gpu}] train ${arm} seed=${seed}"
        PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" -u -m mixture_scaling.train_strict \
          --config "${CONFIG}" --split-root "${SPLIT_ROOT}" \
          --sources "${MIX_SOURCES[$arm]}" --run-id "${run_id}" \
          --device "${gpu}" --seed "${seed}" --output-root "${STATE_ROOT}" \
          --hidden-dim 256 --output-dim 256 >"${LOG_ROOT}/${run_id}.log" 2>&1
      fi
    fi
    ((cursor+=1))
  done
}

pids=(); worker=0
for gpu in 2 3; do
  for slot in 0 1 2; do train_worker "${worker}" "${gpu}" & pids+=("$!"); ((worker+=1)); done
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 )) || exit "${status}"

checkpoint_dir() {
  local arm="$1" seed="$2"
  if [[ "${arm}" == "election" && "${seed}" == 0 ]]; then
    echo "${ROOT}/state/edgeval_ladder_s0/pubmed_k1_subset1_existing_s0/checkpoints"
  elif [[ "${arm}" == "cora" && "${seed}" == 0 ]]; then
    echo "${ROOT}/state/edgeval_ladder_s0/pubmed_k1_subset2_existing_s0/checkpoints"
  elif [[ "${arm}" == "election" ]]; then
    echo "${ROOT}/state/corrected_single_source_matrix_s${seed}/matrix_source_election2020_w256_existing_s${seed}/checkpoints"
  elif [[ "${arm}" == "cora" ]]; then
    echo "${ROOT}/state/corrected_single_source_matrix_s${seed}/matrix_source_cora_w256_existing_s${seed}/checkpoints"
  else
    echo "${STATE_ROOT}/${arm}_existing_s${seed}/checkpoints"
  fi
}

JOBS=()
for arm in election cora mixture_k3 mixture_k6; do
  for seed in 0 1 2; do
    directory="$(checkpoint_dir "${arm}" "${seed}")"
    [[ -d "${directory}" ]] || { echo "missing checkpoint directory ${directory}" >&2; exit 2; }
    while IFS= read -r checkpoint; do JOBS+=("${arm}|${seed}|${checkpoint}"); done \
      < <(find "${directory}" -maxdepth 1 -type f -name 'step_*.pt' | sort -t_ -k2,2n)
  done
done

eval_worker() {
  local index="$1" gpu="$2" cursor=0 job arm seed checkpoint step output
  for job in "${JOBS[@]}"; do
    if (( cursor % 8 == index )); then
      IFS='|' read -r arm seed checkpoint <<<"${job}"
      step="$(basename "${checkpoint}" .pt)"; step="${step#step_}"
      output="${RESULT_ROOT}/raw/${arm}_s${seed}_step_${step}.json"
      if [[ ! -f "${output}" ]]; then
        echo "[gpu ${gpu}] probe ${arm} seed=${seed} step=${step}"
        PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" -u -m mixture_scaling.probe_strict \
          --config "${CONFIG}" --split-root "${SPLIT_ROOT}" --checkpoint "${checkpoint}" \
          --target twibot20 --device "${gpu}" --seed "${seed}" --selection-metric auc \
          --output "${output}" >"${LOG_ROOT}/${arm}_s${seed}_step_${step}.log" 2>&1
      fi
    fi
    ((cursor+=1))
  done
}

pids=(); worker=0
for gpu in 2 3; do
  for slot in 0 1 2 3; do eval_worker "${worker}" "${gpu}" & pids+=("$!"); ((worker+=1)); done
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 )) || exit "${status}"

PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" -m mixture_scaling.aggregate_saturation \
  --raw-root "${RESULT_ROOT}/raw" --output "${RESULT_ROOT}/saturation.csv"
