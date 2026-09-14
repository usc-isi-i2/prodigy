#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/mhchu/miniconda3/envs/prodigy/bin/python3}"
CONFIG="${ROOT}/configs/twibot_strict_pilot.yaml"
SPLIT_ROOT="${ROOT}/state/twibot_strict_pilot/splits"
GRAPHS=(covid_political ukr_rus_suspended election2020 twibot20 facebook_page_reference cora pubmed)
SEEDS=(${SEEDS:-1 2})
TRAIN_WORKERS_PER_GPU="${TRAIN_WORKERS_PER_GPU:-3}"
EVAL_WORKERS_PER_GPU="${EVAL_WORKERS_PER_GPU:-4}"

run_train_worker() {
  local worker_index="$1" gpu="$2" total_workers="$3" index=0 seed graph run_id state_root log_root
  for seed in "${SEEDS[@]}"; do
    state_root="${ROOT}/state/corrected_single_source_matrix_s${seed}"
    log_root="${ROOT}/log/corrected_single_source_matrix_s${seed}"
    mkdir -p "${state_root}" "${log_root}"
    for graph in "${GRAPHS[@]}"; do
      if (( index % total_workers == worker_index )); then
        run_id="matrix_source_${graph}_w256_existing_s${seed}"
        if [[ ! -f "${state_root}/${run_id}/summary.json" ]]; then
          echo "[gpu ${gpu}] train seed=${seed} source=${graph}"
          PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" -u -m mixture_scaling.train_strict \
            --config "${CONFIG}" --split-root "${SPLIT_ROOT}" --sources "${graph}" \
            --run-id "${run_id}" --device "${gpu}" --seed "${seed}" \
            --output-root "${state_root}" --hidden-dim 256 --output-dim 256 \
            >"${log_root}/${run_id}.log" 2>&1
        fi
      fi
      ((index+=1))
    done
  done
}

run_eval_worker() {
  local seed="$1" worker_index="$2" gpu="$3" total_workers="$4"
  local index=0 source target run_id checkpoint output state_root result_root log_root
  state_root="${ROOT}/state/corrected_single_source_matrix_s${seed}"
  result_root="${ROOT}/results/corrected_single_source_matrix_s${seed}"
  log_root="${ROOT}/log/corrected_single_source_matrix_s${seed}"
  mkdir -p "${result_root}/raw" "${log_root}"
  for source in "${GRAPHS[@]}"; do
    run_id="matrix_source_${source}_w256_existing_s${seed}"
    checkpoint="${state_root}/${run_id}/best.pt"
    [[ -f "${checkpoint}" ]] || { echo "missing checkpoint ${checkpoint}" >&2; return 2; }
    for target in "${GRAPHS[@]}"; do
      if (( index % total_workers == worker_index )); then
        output="${result_root}/raw/${source}__to__${target}.json"
        if [[ ! -f "${output}" ]]; then
          echo "[gpu ${gpu}] probe seed=${seed} ${source} -> ${target}"
          PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" -u -m mixture_scaling.probe_strict \
            --config "${CONFIG}" --split-root "${SPLIT_ROOT}" --checkpoint "${checkpoint}" \
            --target "${target}" --device "${gpu}" --seed "${seed}" \
            --selection-metric auc --output "${output}" \
            >"${log_root}/${source}__to__${target}.log" 2>&1
        fi
      fi
      ((index+=1))
    done
  done
}

train_total=$((2 * TRAIN_WORKERS_PER_GPU))
pids=(); worker_index=0
for gpu in 2 3; do
  for ((slot=0; slot<TRAIN_WORKERS_PER_GPU; slot++)); do
    run_train_worker "${worker_index}" "${gpu}" "${train_total}" & pids+=("$!")
    ((worker_index+=1))
  done
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 )) || exit "${status}"

eval_total=$((2 * EVAL_WORKERS_PER_GPU))
pids=(); worker_index=0
for gpu in 2 3; do
  for ((slot=0; slot<EVAL_WORKERS_PER_GPU; slot++)); do
    for seed in "${SEEDS[@]}"; do
      run_eval_worker "${seed}" "${worker_index}" "${gpu}" "${eval_total}" & pids+=("$!")
    done
    ((worker_index+=1))
  done
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 )) || exit "${status}"

for seed in "${SEEDS[@]}"; do
  PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" -m mixture_scaling.aggregate_corrected_matrix \
    --raw-root "${ROOT}/results/corrected_single_source_matrix_s${seed}/raw" \
    --output "${ROOT}/results/corrected_single_source_matrix_s${seed}/matrix.csv" \
    --expected-seed "${seed}"
done
