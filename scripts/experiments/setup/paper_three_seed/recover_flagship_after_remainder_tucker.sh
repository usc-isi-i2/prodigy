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
    write_status waiting "archived failed flagship attempt; waiting for GPUs 0,2,3"
  fi
fi

if [[ ! -e "$REPLICATE_RUN" ]]; then
  stable=0
  while (( stable < 4 )); do
    if nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 0,2,3 |
        awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1>1000 || $2>10) bad=1} END{exit bad}'; then
      stable=$((stable + 1))
    else
      stable=0
    fi
    sleep 30
  done
  write_status training "launching corrected three-seed flagship replicas"
  RUN_ROOT="$FLAGSHIP_ROOT" SEEDS="1 2" GPUS="0 2 3" MODELS_PER_GPU=14 WORKER_BUDGET=168 \
    bash "$SCRIPT_DIR/run_flagship_ladders_tucker.sh"
fi

# VISION owns GPU 1 independently. The fixed evaluations require all four devices.
while tmux has-session -t vision-mixture-seeds 2>/dev/null; do sleep 30; done
stable=0
while (( stable < 4 )); do
  if nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i 0,1,2,3 |
      awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1>1000 || $2>10) bad=1} END{exit bad}'; then
    stable=$((stable + 1))
  else
    stable=0
  fi
  sleep 30
done

if [[ ! -f "$NM_OUTPUT/status.json" ]] || ! grep -q '"status": "complete"' "$NM_OUTPUT/status.json"; then
  write_status evaluating "running fixed 512-episode NM receiver panel"
  "${CONDA_PREFIX}/bin/python" -u scripts/experiments/setup/nm_interventions_overnight/evaluate.py \
    --run-dirs "$REPLICATE_RUN" --output "$NM_OUTPUT" \
    --gpus 0 1 2 3 --workers-per-gpu 2 --episodes 512
fi
if [[ ! -f "$CLS_OUTPUT/classification_long.tsv" ]]; then
  write_status evaluating "running fixed downstream classification panel"
  RUN_STAMP="$RUN_STAMP" OUTPUT_ROOT="$CLS_OUTPUT" REPLICATE_RUN="$REPLICATE_RUN" \
    SEED0_RUN="$SEED0_RUN" GPUS="0 1 2 3" WORKERS_PER_GPU=2 \
    bash "$SCRIPT_DIR/run_flagship_cls_tucker.sh"
fi
write_status complete "corrected flagship training and both evaluations are complete"
