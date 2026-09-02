#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CHECKPOINT="${CHECKPOINT:-/dataMeR1/phil/gfm/prodigy/state/merged_ukr_rus_covid_nm_11_06_2026_18_03_41/checkpoint/state_dict_110000.ckpt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/log/nm_model_size_eval}"
GPU="${GPU:-2}"
REFERENCE_RESULTS="${REFERENCE_RESULTS:-}"
DRY_RUN="${DRY_RUN:-0}"

CONDA_BIN_DIR="${CONDA_BIN_DIR:-/home/mhchu/miniconda3/bin}"
export PATH="${CONDA_BIN_DIR}:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"

mkdir -p "$OUTPUT_ROOT/results" "$OUTPUT_ROOT/runs" "$OUTPUT_ROOT/eval_state"
cd "$REPO_ROOT"

cmd=(
  "$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy
  --config scripts/experiments/setup/covid_ukr/merged_ukr_rus_covid_nm_larger.yaml
  --state-root /dataMeR1/phil/gfm/prodigy/state
  --eval-state-root "$OUTPUT_ROOT/eval_state"
  --log-root "$OUTPUT_ROOT/runs"
  --results "$OUTPUT_ROOT/results/big_7p50m.jsonl"
  --checkpoint-path "$CHECKPOINT"
  --custom-model-id merged_ukr_rus_covid_big_7p50m
  --custom-sources ukr_rus,covid,midterm
  --checkpoint-step 110000
  --training-seeds 0
  --device 0
  --eval-episode-seed-offset 0
)
if [[ -n "$REFERENCE_RESULTS" ]]; then
  cmd+=(--reference-results "$REFERENCE_RESULTS")
fi

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q' "$GPU"
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}"
