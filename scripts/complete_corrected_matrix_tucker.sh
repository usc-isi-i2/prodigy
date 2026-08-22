#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/mhchu/miniconda3/envs/prodigy/bin/python3}"
CONFIG="${ROOT}/configs/twibot_strict_pilot.yaml"
SPLIT_ROOT="${ROOT}/state/twibot_strict_pilot/splits"
STATE_ROOT="${ROOT}/state/corrected_single_source_matrix_s0"
RESULT_ROOT="${ROOT}/results/corrected_single_source_matrix_s0"
LOG_ROOT="${ROOT}/log/corrected_single_source_matrix_s0"
mkdir -p "${STATE_ROOT}" "${RESULT_ROOT}/raw" "${LOG_ROOT}"

FACEBOOK_RUN="matrix_source_facebook_w256_existing_s0"
if [[ ! -f "${STATE_ROOT}/${FACEBOOK_RUN}/summary.json" ]]; then
  PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" -u -m mixture_scaling.train_strict \
    --config "${CONFIG}" --split-root "${SPLIT_ROOT}" \
    --sources facebook_page_reference --run-id "${FACEBOOK_RUN}" \
    --device 2 --seed 0 --output-root "${STATE_ROOT}" \
    --hidden-dim 256 --output-dim 256 >"${LOG_ROOT}/${FACEBOOK_RUN}.log" 2>&1
fi

declare -A CKPT
CKPT[covid_political]="${ROOT}/state/model_scale_edgeval_s0/mscale_election2020_k1_subset0_w256_existing_s0/best.pt"
CKPT[ukr_rus_suspended]="${ROOT}/state/edgeval_ladder_s0/covid_political_k1_subset0_existing_s0/best.pt"
CKPT[election2020]="${ROOT}/state/edgeval_ladder_s0/pubmed_k1_subset1_existing_s0/best.pt"
CKPT[twibot20]="${ROOT}/state/model_scale_edgeval_s0/mscale_election2020_k1_subset1_w256_existing_s0/best.pt"
CKPT[facebook_page_reference]="${STATE_ROOT}/${FACEBOOK_RUN}/best.pt"
CKPT[cora]="${ROOT}/state/edgeval_ladder_s0/pubmed_k1_subset2_existing_s0/best.pt"
CKPT[pubmed]="${ROOT}/state/model_scale_edgeval_s0/mscale_election2020_k1_subset2_w256_existing_s0/best.pt"
GRAPHS=(covid_political ukr_rus_suspended election2020 twibot20 facebook_page_reference cora pubmed)
JOBS=()
for source in "${GRAPHS[@]}"; do
  [[ -f "${CKPT[$source]}" ]] || { echo "missing checkpoint ${CKPT[$source]}" >&2; exit 2; }
  for target in "${GRAPHS[@]}"; do JOBS+=("${source}"$'\t'"${target}"); done
done

WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
TOTAL_WORKERS=$((2 * WORKERS_PER_GPU))
worker() {
  local worker_index="$1" gpu="$2" index=0 job source target output
  for job in "${JOBS[@]}"; do
    if (( index % TOTAL_WORKERS == worker_index )); then
      IFS=$'\t' read -r source target <<< "${job}"
      output="${RESULT_ROOT}/raw/${source}__to__${target}.json"
      if [[ ! -f "${output}" ]]; then
        echo "[gpu ${gpu}] ${source} -> ${target}"
        PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" -u -m mixture_scaling.probe_strict \
          --config "${CONFIG}" --split-root "${SPLIT_ROOT}" \
          --checkpoint "${CKPT[$source]}" --target "${target}" --device "${gpu}" \
          --seed 0 --selection-metric auc --output "${output}" \
          >"${LOG_ROOT}/${source}__to__${target}.log" 2>&1
      fi
    fi
    ((index+=1))
  done
}

pids=(); worker_index=0
for gpu in 2 3; do
  for ((slot=0; slot<WORKERS_PER_GPU; slot++)); do
    worker "${worker_index}" "${gpu}" & pids+=("$!"); ((worker_index+=1))
  done
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 )) || exit "${status}"

PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" -m mixture_scaling.aggregate_corrected_matrix \
  --raw-root "${RESULT_ROOT}/raw" --output "${RESULT_ROOT}/matrix.csv"
