#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/mhchu/miniconda3/envs/prodigy/bin/python3}"
EVAL_WORKERS_PER_GPU="${EVAL_WORKERS_PER_GPU:-${WORKERS_PER_GPU:-4}}"
TRAIN_WORKERS_PER_GPU="${TRAIN_WORKERS_PER_GPU:-2}"
PIPELINE_LOG_ROOT="${PIPELINE_LOG_ROOT:-${ROOT}/log/pipeline_followups}"
mkdir -p "${PIPELINE_LOG_ROOT}"
[[ -f "${ROOT}/results/primary_s0/primary_results.csv" ]] || {
  echo "primary aggregate is missing" >&2; exit 1;
}

export PATH="/home/mhchu/miniconda3/bin:${PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${ROOT}"

echo "full seed-0 transfer matrix" | tee "${PIPELINE_LOG_ROOT}/status.txt"
MANIFEST="${ROOT}/manifests/matrix_s0_eval.tsv" \
STATE_ROOT="${ROOT}/state/primary_s0" \
RESULT_ROOT="${ROOT}/results/matrix_s0/raw" \
LOG_ROOT="${ROOT}/log/matrix_s0_eval" GPUS="2 3" WORKERS_PER_GPU="${EVAL_WORKERS_PER_GPU}" \
  bash "${ROOT}/scripts/eval_manifest_tucker.sh" >"${PIPELINE_LOG_ROOT}/matrix.log" 2>&1
PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" -m mixture_scaling.aggregate \
  --raw-root "${ROOT}/results/matrix_s0/raw" \
  --output "${ROOT}/results/matrix_s0/matrix_results.csv" --expected 392

echo "intermediate seed-0 mixtures" | tee "${PIPELINE_LOG_ROOT}/status.txt"
MANIFEST="${ROOT}/manifests/ladder_s0_train.tsv" \
STATE_ROOT="${ROOT}/state/ladder_s0" LOG_ROOT="${ROOT}/log/ladder_s0" GPUS="2 3" WORKERS_PER_GPU="${TRAIN_WORKERS_PER_GPU}" \
  bash "${ROOT}/scripts/run_manifest_tucker.sh" >"${PIPELINE_LOG_ROOT}/ladder_train.log" 2>&1
MANIFEST="${ROOT}/manifests/ladder_s0_eval.tsv" \
STATE_ROOT="${ROOT}/state/ladder_s0" \
RESULT_ROOT="${ROOT}/results/ladder_s0/raw" \
LOG_ROOT="${ROOT}/log/ladder_s0_eval" GPUS="2 3" WORKERS_PER_GPU="${EVAL_WORKERS_PER_GPU}" \
  bash "${ROOT}/scripts/eval_manifest_tucker.sh" >"${PIPELINE_LOG_ROOT}/ladder_eval.log" 2>&1
PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" -m mixture_scaling.aggregate \
  --raw-root "${ROOT}/results/ladder_s0/raw" \
  --output "${ROOT}/results/ladder_s0/ladder_results.csv" --expected 112

echo "primary replications seeds 1 and 2" | tee "${PIPELINE_LOG_ROOT}/status.txt"
MANIFEST="${ROOT}/manifests/primary_s1_s2.tsv" \
STATE_ROOT="${ROOT}/state/primary_s1_s2" LOG_ROOT="${ROOT}/log/primary_s1_s2" GPUS="2 3" WORKERS_PER_GPU="${TRAIN_WORKERS_PER_GPU}" \
  bash "${ROOT}/scripts/run_manifest_tucker.sh" >"${PIPELINE_LOG_ROOT}/seeds_train.log" 2>&1
MANIFEST="${ROOT}/manifests/primary_s1_s2.tsv" \
STATE_ROOT="${ROOT}/state/primary_s1_s2" \
RESULT_ROOT="${ROOT}/results/primary_s1_s2/raw" \
LOG_ROOT="${ROOT}/log/primary_s1_s2_eval" GPUS="2 3" WORKERS_PER_GPU="${EVAL_WORKERS_PER_GPU}" \
  bash "${ROOT}/scripts/eval_manifest_tucker.sh" >"${PIPELINE_LOG_ROOT}/seeds_eval.log" 2>&1
PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" -m mixture_scaling.aggregate \
  --raw-root "${ROOT}/results/primary_s1_s2/raw" \
  --output "${ROOT}/results/primary_s1_s2/primary_results.csv" --expected 112

echo "complete $(date -u +%FT%TZ)" | tee "${PIPELINE_LOG_ROOT}/status.txt"
