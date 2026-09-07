#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PAIR_WORKTREE="${PAIR_WORKTREE:-/dataMeR1/phil/gfm/prodigy-nm-pairs}"
PAIR_TRAIN_STATUS="${PAIR_TRAIN_STATUS:-${PAIR_WORKTREE}/log/nm_pairwise_finalcore/shared_seed0_20260904/status.json}"
PAIR_EVAL_COMPLETE="${PAIR_EVAL_COMPLETE:-${PAIR_WORKTREE}/log/nm_pairwise_finalcore_eval/production/bs32/complete_utc.txt}"
PIPELINE_LOG_ROOT="${PIPELINE_LOG_ROOT:-${REPO_ROOT}/log/nm_leave_one_out_finalcore_pipeline}"
MAX_EXISTING_GPU_MIB="${MAX_EXISTING_GPU_MIB:-1000}"

mkdir -p "$PIPELINE_LOG_ROOT"
cd "$REPO_ROOT"

echo "waiting for strict pair-evaluation receipt: $PAIR_EVAL_COMPLETE"
while [[ ! -f "$PAIR_EVAL_COMPLETE" ]]; do
  if [[ -f "$PAIR_TRAIN_STATUS" ]]; then
    state="$(/home/mhchu/miniconda3/envs/prodigy/bin/python -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("status", "unknown"))' \
      "$PAIR_TRAIN_STATUS")"
    [[ "$state" != failed_or_interrupted ]] || {
      echo "pair training failed; LOO pipeline will not start" >&2
      exit 1
    }
    if [[ "$state" == complete ]] && ! tmux has-session -t nmpairs-followon 2>/dev/null; then
      echo "pair evaluation session ended without a completion receipt" >&2
      exit 1
    fi
  elif ! tmux has-session -t nmpairs 2>/dev/null; then
    echo "pair training session ended without a status receipt" >&2
    exit 1
  fi
  sleep 30
done

echo "pair evaluation complete at $(cat "$PAIR_EVAL_COMPLETE")"
while true; do
  busy=0
  for gpu in 0 1 2 3; do
    used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    (( used <= MAX_EXISTING_GPU_MIB )) || busy=1
  done
  (( busy == 1 )) || break
  echo "waiting for pair evaluation to release GPUs 0--3"
  sleep 15
done

echo "starting LOO training at $(date -u +%FT%TZ)"
bash "$SCRIPT_DIR/run_training_tucker.sh" \
  > "$PIPELINE_LOG_ROOT/training.log" 2>&1

"/home/mhchu/miniconda3/envs/prodigy/bin/python" "$SCRIPT_DIR/verify_training.py" \
  --run-dir "$REPO_ROOT/log/nm_leave_one_out_finalcore/shared_seed0_20260904" \
  > "$PIPELINE_LOG_ROOT/training_verification.json"

echo "starting held-out LOO evaluation at $(date -u +%FT%TZ)"
bash "$SCRIPT_DIR/run_evaluation_tucker.sh" \
  > "$PIPELINE_LOG_ROOT/evaluation.log" 2>&1

date -u +%FT%TZ > "$PIPELINE_LOG_ROOT/complete_utc.txt"
echo "LOO pipeline complete at $(cat "$PIPELINE_LOG_ROOT/complete_utc.txt")"
