#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BASE_STATE="${BASE_STATE:-/dataMeR1/phil/gfm/prodigy-rq1/state/rq1_label_efficiency_loto}"
BASE_LOG="${BASE_LOG:-${REPO_ROOT}/log/rq1_label_efficiency_loto}"
PRETRAIN_ROOT="${BASE_STATE}/pretrain"
ADAPT_ROOT="${BASE_STATE}/adapt_cached_v2"
CACHE_ROOT="${BASE_STATE}/subgraph_cache_v2"
RUN_STAMP="${RUN_STAMP:-20260828}"

# The prior worktree is still completing the four seed-1 pretraining models.
# Wait for all selected checkpoints and for their trainers to exit before taking over.
while :; do
  completed=0
  for target in covid_political election2020 ukr_rus_suspended twibot20; do
    checkpoint="${PRETRAIN_ROOT}/rq1_loto_${target}_pretrain_s1_${RUN_STAMP}/state_dict"
    [[ -f "$checkpoint" ]] && ((completed+=1))
  done
  active="$(pgrep -af 'run_single_experiment.py.*pretrain_s1' | grep -v grep | wc -l | tr -d ' ' || true)"
  echo "seed1 pretraining checkpoints=${completed}/4 active=${active}"
  [[ "$completed" -eq 4 && "$active" -eq 0 ]] && break
  sleep 300
done

# Its parent was deliberately stopped before seed 1 adaptation; discard only the
# stale tmux process tree after all child trainers have cleanly completed.
tmux kill-session -t rq1_all 2>/dev/null || true

SEED=1 PRETRAIN_STATE_ROOT="$PRETRAIN_ROOT" OUTPUT_ROOT="$ADAPT_ROOT" \
  LOG_ROOT="${BASE_LOG}/adapt_cached_v2" CACHE_ROOT="$CACHE_ROOT" \
  bash "${SCRIPT_DIR}/run_adapt_seed_tucker.sh"

# Experimental third seed: revised validation cadence/stopping, fully isolated.
BASE_STATE="$BASE_STATE" BASE_LOG="$BASE_LOG" CACHE_ROOT="$CACHE_ROOT" \
  bash "${SCRIPT_DIR}/run_seed2_revised_tucker.sh"

# Paired canonical third seed, preserving the old protocol and final RQ1 namespace.
SEED=2 CONFIG="${SCRIPT_DIR}/pretrain.yaml" STATE_ROOT="$PRETRAIN_ROOT" \
  LOG_ROOT="${BASE_LOG}/pretrain" bash "${SCRIPT_DIR}/run_pretrain_seed_tucker.sh"
SEED=2 PRETRAIN_STATE_ROOT="$PRETRAIN_ROOT" OUTPUT_ROOT="$ADAPT_ROOT" \
  LOG_ROOT="${BASE_LOG}/adapt_cached_v2" CACHE_ROOT="$CACHE_ROOT" \
  bash "${SCRIPT_DIR}/run_adapt_seed_tucker.sh"
