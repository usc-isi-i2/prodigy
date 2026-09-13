#!/usr/bin/env bash
# Evaluate all optimized core replicas on the campaign's GPUs 2--3.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RUN_STAMP="${RUN_STAMP:-20260908}"
SEED1_ONEHOP="${SEED1_ONEHOP:-/dataMeR1/phil/gfm/prodigy-paper-fast/log/paper_three_seed_fast/${RUN_STAMP}/seed1_ladder_1hop}"
REMAINDER_ROOT="${REMAINDER_ROOT:-/dataMeR1/phil/gfm/prodigy-paper-remainder-opt/log/paper_three_seed_remainder/${RUN_STAMP}}"
SEED2_ONEHOP="${SEED2_ONEHOP:-${REMAINDER_ROOT}/onehop_seed_2}"
TWOHOP="${TWOHOP:-${REMAINDER_ROOT}/twohop_seeds_1-2}"
OUTPUT="${OUTPUT:-${REMAINDER_ROOT}/core_nm_evaluation}"
WAIT_SESSION="${WAIT_SESSION:-paper-optimized-queue}"
GPUS_TEXT="${GPUS:-}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
unset CUDA_VISIBLE_DEVICES || true
cd "$REPO_ROOT"
mkdir -p log

wait_complete() {
  local run_dir="$1" expected="$2"
  while [[ ! -f "$run_dir/status.json" ]]; do sleep 60; done
  "${CONDA_PREFIX}/bin/python" - "$run_dir/status.json" "$expected" <<'PY'
import json
import sys
path, expected_text = sys.argv[1:]
status = json.load(open(path, encoding="utf-8"))
finished = status.get("finished", [])
if status.get("status") != "complete" or len(finished) != int(expected_text):
    raise SystemExit(f"invalid upstream status: {path}")
if any(row.get("exitcode") != 0 for row in finished):
    raise SystemExit(f"nonzero upstream exit: {path}")
PY
}

wait_complete "$SEED1_ONEHOP" 18
wait_complete "$SEED2_ONEHOP" 18
wait_complete "$TWOHOP" 46

while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do sleep 60; done

if [[ -z "$GPUS_TEXT" ]]; then
  while true; do
    available=()
    for gpu in 2 3; do
      values="$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits | tr -d ' ')"
      IFS=, read -r used util <<< "$values"
      if (( used < 1000 && util < 10 )); then available+=("$gpu"); fi
    done
    if (( ${#available[@]} >= 2 )); then
      GPUS_TEXT="${available[*]}"
      break
    fi
    sleep 30
  done
fi
read -r -a gpu_ids <<< "$GPUS_TEXT"
(( ${#gpu_ids[@]} >= 2 )) || { echo "core evaluation requires at least two GPUs" >&2; exit 2; }
for gpu in "${gpu_ids[@]}"; do
  [[ "$gpu" =~ ^[23]$ ]] || { echo "refusing GPU $gpu: this campaign is restricted to 2-3" >&2; exit 2; }
done
[[ "${gpu_ids[*]}" == "2 3" || "${gpu_ids[*]}" == "3 2" ]] \
  || { echo "core evaluation requires exactly GPUs 2 and 3" >&2; exit 2; }
stable=0
while (( stable < 4 )); do
  gpu_csv="$(tr ' ' ',' <<< "$GPUS_TEXT")"
  if nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu_csv" |
      awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1>1000 || $2>10) bad=1} END{exit bad}'; then
    stable=$((stable + 1))
  else
    stable=0
  fi
  sleep 30
done

"${CONDA_PREFIX}/bin/python" -u "$SCRIPT_DIR/evaluate_fast_core.py" \
  --run-group "onehop=$SEED1_ONEHOP" \
  --run-group "onehop=$SEED2_ONEHOP" \
  --run-group "twohop=$TWOHOP" \
  --output "$OUTPUT" \
  --eval-config scripts/experiments/setup/final_core/training.yaml \
  --gpus $GPUS_TEXT --workers-per-gpu 2 \
  --episodes 512 --expected-models 82
date -u +%Y-%m-%dT%H:%M:%SZ > "$OUTPUT/complete_utc.txt"
