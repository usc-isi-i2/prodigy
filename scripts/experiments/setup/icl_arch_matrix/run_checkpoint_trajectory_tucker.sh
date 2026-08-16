#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
STATE_ROOT="${STATE_ROOT:-/dataMeR1/phil/gfm/worktree-runtime-archive-20260812/prodigy-archmatrix/files/state/icl_arch_matrix}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/icl_arch_matrix/self_trajectory}"
VISION_ROOT="${VISION_ROOT:-/dataMeR1/phil/gfm/upstream/VISION}"
GILT_ROOT="${GILT_ROOT:-/dataMeR1/phil/gfm/upstream/inductnode}"
RUN_STAMP="${RUN_STAMP:-20260810}"
CHECKPOINT_STEPS_TEXT="${CHECKPOINT_STEPS:-20 60}"
MODEL_IDS_TEXT="${MODEL_IDS:-ss_covid_political ss_election2020 ss_ukr_rus_suspended ss_twibot20}"
DRY_RUN="${DRY_RUN:-0}"
read -r -a CHECKPOINT_STEPS_ARRAY <<< "$CHECKPOINT_STEPS_TEXT"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"
mkdir -p "$LOG_ROOT/results" "$LOG_ROOT/queue" "$LOG_ROOT/prodigy_state"
cd "$REPO_ROOT"

model_ids_csv="${MODEL_IDS_TEXT// /,}"
for step in "${CHECKPOINT_STEPS_ARRAY[@]}"; do
  [[ "$step" == 20 || "$step" == 60 || "$step" == 100 ]] || {
    echo "unsupported checkpoint step: $step" >&2
    exit 2
  }
  common=(--state-root "$STATE_ROOT" --checkpoint-step "$step" --model-ids "$model_ids_csv")
  prodigy_cmd=("$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy
    "${common[@]}" --log-root "$LOG_ROOT/prodigy_runs_step${step}"
    --eval-state-root "$LOG_ROOT/prodigy_state" --results "$LOG_ROOT/results/prodigy_step${step}.jsonl"
    --run-stamp "$RUN_STAMP" --device 0)
  vision_cmd=("$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.evaluate_adapters
    --architecture vision --upstream-root "$VISION_ROOT" "${common[@]}"
    --results "$LOG_ROOT/results/vision_step${step}.jsonl" --device 0)
  gilt_cmd=("$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.evaluate_adapters
    --architecture gilt --upstream-root "$GILT_ROOT" "${common[@]}"
    --results "$LOG_ROOT/results/gilt_step${step}.jsonl" --device 0)

  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'DRY CUDA_VISIBLE_DEVICES=0'; printf ' %q' "${prodigy_cmd[@]}"; printf '\n'
    printf 'DRY CUDA_VISIBLE_DEVICES=1'; printf ' %q' "${vision_cmd[@]}"; printf '\n'
    printf 'DRY CUDA_VISIBLE_DEVICES=0'; printf ' %q' "${gilt_cmd[@]}"; printf '\n'
    continue
  fi

  CUDA_VISIBLE_DEVICES=0 "${prodigy_cmd[@]}" > "$LOG_ROOT/queue/prodigy_step${step}.log" 2>&1 & p0=$!
  CUDA_VISIBLE_DEVICES=1 "${vision_cmd[@]}" > "$LOG_ROOT/queue/vision_step${step}.log" 2>&1 & p1=$!
  status=0
  wait "$p0" || status=1
  wait "$p1" || status=1
  (( status == 0 )) || exit "$status"
  CUDA_VISIBLE_DEVICES=0 "${gilt_cmd[@]}" > "$LOG_ROOT/queue/gilt_step${step}.log" 2>&1
done
