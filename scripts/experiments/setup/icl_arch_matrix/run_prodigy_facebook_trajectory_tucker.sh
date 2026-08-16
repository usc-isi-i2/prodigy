#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
STATE_ROOT="${STATE_ROOT:-/dataMeR1/phil/gfm/worktree-runtime-archive-20260812/prodigy-archmatrix/files/state/icl_arch_matrix}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/icl_arch_matrix/prodigy_facebook_trajectory}"
RUN_STAMP="${RUN_STAMP:-20260810}"
CHECKPOINT_STEPS_TEXT="${CHECKPOINT_STEPS:-20 60 100}"
MODEL_IDS_TEXT="${MODEL_IDS:-ss_covid_political ss_election2020 ss_ukr_rus_suspended ss_twibot20}"
DATASETS_TEXT="${DATASETS:-facebook_page_reference}"
GPU="${GPU:-0}"
DO_RANDOM_INIT="${DO_RANDOM_INIT:-0}"
read -r -a CHECKPOINT_STEPS_ARRAY <<< "$CHECKPOINT_STEPS_TEXT"

[[ "$GPU" == 0 || "$GPU" == 1 ]] || {
  echo "GPU must be 0 or 1, got: $GPU" >&2
  exit 2
}

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source /home/mhchu/miniconda3/etc/profile.d/conda.sh
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"

mkdir -p "$LOG_ROOT/results" "$LOG_ROOT/runs" "$LOG_ROOT/eval_state"
cd "$REPO_ROOT"
model_ids_csv="${MODEL_IDS_TEXT// /,}"

for step in "${CHECKPOINT_STEPS_ARRAY[@]}"; do
  [[ "$step" == 20 || "$step" == 60 || "$step" == 100 ]] || {
    echo "unsupported checkpoint step: $step" >&2
    exit 2
  }
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u -m \
    scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy \
    --state-root "$STATE_ROOT" \
    --checkpoint-step "$step" \
    --model-ids "$model_ids_csv" \
    --datasets "$DATASETS_TEXT" \
    --include-facebook \
    --log-root "$LOG_ROOT/runs/gpu${GPU}_step${step}" \
    --eval-state-root "$LOG_ROOT/eval_state" \
    --results "$LOG_ROOT/results/gpu${GPU}_step${step}.jsonl" \
    --run-stamp "$RUN_STAMP" \
    --device 0
done

if [[ "$DO_RANDOM_INIT" == 1 ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u -m \
    scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy \
    --state-root "$STATE_ROOT" \
    --random-init \
    --datasets facebook_page_reference \
    --include-facebook \
    --log-root "$LOG_ROOT/runs/random_init" \
    --eval-state-root "$LOG_ROOT/eval_state" \
    --results "$LOG_ROOT/results/random_init.jsonl" \
    --run-stamp "$RUN_STAMP" \
    --device 0
fi
