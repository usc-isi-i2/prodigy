#!/usr/bin/env bash
# Recover the flagship handoff from a corrected worktree without mutating live worktrees.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_STAMP="${RUN_STAMP:-20260908}"
SHARED_ROOT="${SHARED_ROOT:-/dataMeR1/phil/gfm/prodigy-paper-remainder-opt/log}"
FLAGSHIP_ROOT="${SHARED_ROOT}/paper_flagship_ladders/${RUN_STAMP}"
REPLICATE_RUN="${FLAGSHIP_ROOT}/seeds_1-2"
NM_OUTPUT="${FLAGSHIP_ROOT}/nm_evaluation"
CLS_OUTPUT="${FLAGSHIP_ROOT}/classification_evaluation"
SEED0_RUN="${SEED0_RUN:-/dataMeR1/phil/gfm/prodigy-nmi-overnight/log/production/train_20260904_023434}"
ONEHOP_STATUS="${SHARED_ROOT}/paper_three_seed_remainder/${RUN_STAMP}/onehop_seed_2/status.json"
TWOHOP_STATUS="${SHARED_ROOT}/paper_three_seed_remainder/${RUN_STAMP}/twohop_seeds_1-2/status.json"
REMAINDER_TRAIN_COMPLETE="${SHARED_ROOT}/paper_three_seed_remainder/${RUN_STAMP}/training_complete_utc.txt"
STATUS_FILE="${REPO_ROOT}/log/paper_flagship_recovery_${RUN_STAMP}.json"
MECHANISM_TRAIN_COMPLETE="${MECHANISM_TRAIN_COMPLETE:-/dataMeR1/phil/gfm/prodigy-paper-mechanism/log/paper_mechanism_sweeps/${RUN_STAMP}/training_complete_utc.txt}"
ALL_PAIRS_TRAIN_COMPLETE="${ALL_PAIRS_TRAIN_COMPLETE:-/dataMeR1/phil/gfm/prodigy-paper-all-pairs/log/paper_all_pairs/${RUN_STAMP}/training_complete_utc.txt}"
TRAIN_GPUS="${TRAIN_GPUS:-}"
EVAL_GPUS="${EVAL_GPUS:-}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset CUDA_VISIBLE_DEVICES || true
cd "$REPO_ROOT"
mkdir -p "$(dirname "$STATUS_FILE")"

write_status() {
  local status="$1" detail="$2"
  STATUS="$status" DETAIL="$detail" PATH_OUT="$STATUS_FILE" "${CONDA_PREFIX}/bin/python" - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path
path = Path(os.environ["PATH_OUT"])
path.write_text(json.dumps({
    "status": os.environ["STATUS"], "detail": os.environ["DETAIL"],
    "updated_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\n")
PY
}
trap 'write_status failed "corrected flagship recovery failed"' ERR

wait_for_marker() {
  local marker="$1" detail="$2"
  shift 2
  while [[ ! -f "$marker" ]]; do
    write_status waiting "$detail"
    local alive=0 session
    for session in "$@"; do
      if tmux has-session -t "$session" 2>/dev/null; then alive=1; fi
    done
    (( alive == 1 )) || { echo "missing producer for $marker" >&2; return 1; }
    sleep 30
  done
}

gpu_idle() {
  local gpu="$1" values used util
  values="$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ' ')"
  IFS=, read -r used util <<< "$values"
  (( used < 1000 && util < 10 ))
}

validate_gpu_list() {
  local text="$1" minimum="$2" gpu
  local -A seen=()
  read -r -a values <<< "$text"
  (( ${#values[@]} >= minimum )) || return 1
  for gpu in "${values[@]}"; do
    [[ "$gpu" =~ ^[0-3]$ ]] || return 1
    [[ -z "${seen[$gpu]:-}" ]] || return 1
    seen[$gpu]=1
  done
}

wait_for_stable_gpus() {
  local text="$1" stable=0 gpu_csv
  validate_gpu_list "$text" 1 || return 1
  gpu_csv="$(tr ' ' ',' <<< "$text")"
  while (( stable < 4 )); do
    if nvidia-smi --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits -i "$gpu_csv" |
        awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1>1000 || $2>10) bad=1} END{exit bad}'; then
      stable=$((stable + 1))
    else
      stable=0
    fi
    sleep 30
  done
}

choose_training_gpus() {
  if [[ -n "$TRAIN_GPUS" ]]; then
    validate_gpu_list "$TRAIN_GPUS" 3 || return 1
    echo "$TRAIN_GPUS"
    return
  fi
  while true; do
    if gpu_idle 0; then
      echo "0 2 3"
      return
    fi
    if ! tmux has-session -t vision-mixture-seeds 2>/dev/null && gpu_idle 1; then
      echo "1 2 3"
      return
    fi
    write_status waiting "waiting for either GPU 0 or post-VISION GPU 1"
    sleep 30
  done
}

choose_evaluation_gpus() {
  if [[ -n "$EVAL_GPUS" ]]; then
    validate_gpu_list "$EVAL_GPUS" 3 || return 1
    echo "$EVAL_GPUS"
    return
  fi
  while true; do
    local available=() gpu
    for gpu in 0 1 2 3; do
      if gpu_idle "$gpu"; then available+=("$gpu"); fi
    done
    if (( ${#available[@]} >= 3 )); then
      echo "${available[*]}"
      return
    fi
    write_status waiting "waiting for at least three idle owned evaluation GPUs"
    sleep 30
  done
}

wait_for_marker "$REMAINDER_TRAIN_COMPLETE" "waiting for remainder recovery" \
  paper-optimized-queue paper-remainder-recovery

# The mechanism queue may deliberately refill the six-per-GPU slots released
# by the one-hop overlap. If the old remainder exits unexpectedly early, do not
# start the flagship until that bounded 10k-update training phase releases 0/2/3.
wait_for_marker "$MECHANISM_TRAIN_COMPLETE" "waiting for mechanism training" \
  paper-mechanism-sweeps paper-mechanism-eval
wait_for_marker "$ALL_PAIRS_TRAIN_COMPLETE" "waiting for all-pairs recovery" \
  paper-all-pairs-after-mechanism paper-all-pairs-recovery

"${CONDA_PREFIX}/bin/python" - "$ONEHOP_STATUS" 18 "$TWOHOP_STATUS" 46 <<'PY'
import json, sys
for path, expected in ((sys.argv[1], int(sys.argv[2])), (sys.argv[3], int(sys.argv[4]))):
    row = json.load(open(path))
    finished = row.get("finished", [])
    if row.get("status") != "complete" or len(finished) != expected or any(x.get("exitcode") != 0 for x in finished):
        raise ValueError(f"invalid prerequisite status: {path}")
PY

if [[ -f "$NM_OUTPUT/status.json" ]] && grep -q '"status": "complete"' "$NM_OUTPUT/status.json" \
    && [[ -f "$CLS_OUTPUT/classification_long.tsv" ]]; then
  write_status complete "existing flagship training and both evaluations are complete"
  exit 0
fi

if [[ -e "$REPLICATE_RUN" ]]; then
  if [[ -f "$REPLICATE_RUN/status.json" ]] && grep -q '"status": "complete"' "$REPLICATE_RUN/status.json"; then
    write_status evaluating "using completed flagship replicas from the prior orchestrator"
  else
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    mv "$REPLICATE_RUN" "${REPLICATE_RUN}.failed_sampler_${stamp}"
    write_status waiting "archived failed flagship attempt; selecting idle owned GPUs"
  fi
fi

if [[ ! -e "$REPLICATE_RUN" ]]; then
  train_gpus="$(choose_training_gpus)"
  wait_for_stable_gpus "$train_gpus"
  read -r -a train_gpu_ids <<< "$train_gpus"
  write_status training "launching corrected flagship replicas on GPUs $train_gpus"
  RUN_ROOT="$FLAGSHIP_ROOT" SEEDS="1 2" GPUS="$train_gpus" MODELS_PER_GPU=14 \
    WORKER_BUDGET="$((${#train_gpu_ids[@]} * 56))" \
    bash "$SCRIPT_DIR/run_flagship_ladders_tucker.sh"
fi

# Evaluation waits for VISION, then uses every idle owned device with a
# three-device minimum. This avoids blocking on an unrelated user of GPU 0.
while tmux has-session -t vision-mixture-seeds 2>/dev/null; do sleep 30; done
eval_gpus="$(choose_evaluation_gpus)"
wait_for_stable_gpus "$eval_gpus"

if [[ ! -f "$NM_OUTPUT/status.json" ]] || ! grep -q '"status": "complete"' "$NM_OUTPUT/status.json"; then
  write_status evaluating "running fixed 512-episode NM receiver panel on GPUs $eval_gpus"
  "${CONDA_PREFIX}/bin/python" -u scripts/experiments/setup/nm_interventions_overnight/evaluate.py \
    --run-dirs "$REPLICATE_RUN" --output "$NM_OUTPUT" \
    --gpus $eval_gpus --workers-per-gpu 2 --episodes 512
fi
if [[ ! -f "$CLS_OUTPUT/classification_long.tsv" ]]; then
  write_status evaluating "running fixed downstream classification panel on GPUs $eval_gpus"
  RUN_STAMP="$RUN_STAMP" OUTPUT_ROOT="$CLS_OUTPUT" REPLICATE_RUN="$REPLICATE_RUN" \
    SEED0_RUN="$SEED0_RUN" GPUS="$eval_gpus" WORKERS_PER_GPU=2 \
    bash "$SCRIPT_DIR/run_flagship_cls_tucker.sh"
fi
write_status complete "corrected flagship training and both evaluations are complete"
