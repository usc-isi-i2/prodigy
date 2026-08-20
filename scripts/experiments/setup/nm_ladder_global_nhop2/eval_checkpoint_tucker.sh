#!/usr/bin/env bash
# Run the exact final-core fixed-test evaluator for one completed treatment model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
FINAL_CORE="${SCRIPT_DIR}/../final_core"
[[ $# -eq 2 ]] || { echo "usage: $0 <final-core physical job index> <gpu>" >&2; exit 2; }
JOB_INDEX="$1"
GPU="$2"
[[ "${GPU}" =~ ^[01]$ ]] || { echo "evaluation is restricted to GPU 0 or 1" >&2; exit 2; }

STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state/nm_ladder_global_finalcore}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/nm_ladder_global_finalcore}"
RUN_STAMP="${RUN_STAMP:-20260814global}"
REFERENCE="${REPO_ROOT}/scripts/experiments/analysis/transfer/matrices/cross_model/final_core/data/prodigy_final_core/auc/reference/episode_fingerprints.tsv"
RESULTS_ROOT="${LOG_ROOT}/fixed_test/results"
READY_DIR="${LOG_ROOT}/fixed_test/ready/job_${JOB_INDEX}"

export PATH="/home/mhchu/miniconda3/bin:${PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
export FINAL_CORE_CPU_THREADS="${FINAL_CORE_CPU_THREADS:-24}"
mkdir -p "${RESULTS_ROOT}" "${READY_DIR}" "${LOG_ROOT}/fixed_test/internal" \
  "${LOG_ROOT}/fixed_test/state"
cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${GPU}" "${CONDA_PREFIX}/bin/python" -u \
  "${FINAL_CORE}/evaluate_fixed_grid.py" \
  --worker-index "${JOB_INDEX}" --worker-count 93 --max-checkpoints 1 \
  --targets "ukr_rus,covid,midterm,covid_political,election2020,ukr_rus_suspended,twibot20,cp_hk,facebook_page_reference" \
  --batch-size 32 --episode-count 512 \
  --config "${FINAL_CORE}/training.yaml" \
  --training-state-root "${STATE_ROOT}" --training-run-stamp "${RUN_STAMP}" \
  --evaluation-state-root "${LOG_ROOT}/fixed_test/state/job_${JOB_INDEX}" \
  --evaluation-log-root "${LOG_ROOT}/fixed_test/internal/job_${JOB_INDEX}" \
  --results-root "${RESULTS_ROOT}" \
  --evaluation-run-stamp "${RUN_STAMP}_job${JOB_INDEX}" \
  --ready-dir "${READY_DIR}" --expected-workers 1 \
  --reference-fingerprints "${REFERENCE}"
