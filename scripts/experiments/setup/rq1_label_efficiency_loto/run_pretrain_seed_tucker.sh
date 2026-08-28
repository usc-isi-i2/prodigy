#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/pretrain.yaml}"
STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state/rq1_label_efficiency_loto/pretrain}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/rq1_label_efficiency_loto/pretrain}"
SEED="${SEED:?set SEED to 0, 1, or 2}"
RUN_STAMP="${RUN_STAMP:-20260828}"
GPUS_TEXT="${GPUS_TEXT:-2 3}"
DRY_RUN="${DRY_RUN:-0}"
read -r -a GPUS <<< "$GPUS_TEXT"
[[ "$SEED" =~ ^[012]$ ]] || { echo "SEED must be 0, 1, or 2" >&2; exit 2; }
[[ "${#GPUS[@]}" -eq 2 ]] || { echo "exactly two GPUs required" >&2; exit 2; }
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[23]$ ]] || { echo "only Tucker GPUs 2 and 3 are authorized" >&2; exit 2; }
done

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
PYTHON="${CONDA_PREFIX}/bin/python"
mkdir -p "$STATE_ROOT" "$LOG_ROOT/train" "$LOG_ROOT/manifest"
cd "$REPO_ROOT"

mapfile -t jobs < <("$PYTHON" "$SCRIPT_DIR/plan.py" | tail -n +2)

run_target() {
  local worker="$1" gpu="$2" row target excluded sources prefix run_name run_dir best latest resume_arg
  for index in "${!jobs[@]}"; do
    (( index % 2 == worker )) || continue
    row="${jobs[$index]}"
    IFS=$'\t' read -r target excluded sources <<< "$row"
    prefix="rq1_loto_${target}_pretrain_s${SEED}"
    run_name="${prefix}_${RUN_STAMP}"
    run_dir="$STATE_ROOT/$run_name"
    best="$run_dir/state_dict"
    if [[ -f "$best" ]]; then
      echo "[gpu $gpu] SKIP completed $run_name"
      continue
    fi
    resume_arg=()
    if [[ -d "$run_dir/checkpoint" ]]; then
      latest="$(find "$run_dir/checkpoint" -type f -name 'training_state_*.ckpt' -print | sort -V | tail -n 1)"
      [[ -n "$latest" ]] && resume_arg=(--resume_training_checkpoint "$latest")
    fi
    cmd=("$PYTHON" -u experiments/run_single_experiment.py --config "$CONFIG"
      --device "$gpu" --seed "$SEED" --prefix "$prefix" --timestamp "$RUN_STAMP"
      --state_dir "$STATE_ROOT" --log_dir "$LOG_ROOT"
      --neighbor_sampling_source_subset "$sources" "${resume_arg[@]}")
    printf '[gpu %s] target=%s excluded=%s sources=%s cmd=' "$gpu" "$target" "$excluded" "$sources"
    printf '%q ' "${cmd[@]}"; printf '\n'
    [[ "$DRY_RUN" == 1 ]] && continue
    "${cmd[@]}" > "$LOG_ROOT/train/${run_name}.log" 2>&1
    [[ -f "$best" ]] || { echo "missing best checkpoint: $best" >&2; return 1; }
    "$PYTHON" -c "import json; from pathlib import Path; p=Path('$LOG_ROOT/manifest/${run_name}.json'); p.write_text(json.dumps({'target':'$target','excluded_family':'$excluded'.split(','),'sources':'$sources'.split(','),'seed':$SEED,'checkpoint':'$best'},sort_keys=True)+'\\n')"
  done
}

run_target 0 "${GPUS[0]}" & left=$!
run_target 1 "${GPUS[1]}" & right=$!
status=0
wait "$left" || status=1
wait "$right" || status=1
exit "$status"
