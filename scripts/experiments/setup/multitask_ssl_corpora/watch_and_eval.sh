#!/usr/bin/env bash
# Auto-eval watcher for multitask_ssl_corpora (the tfssl-E2 pattern). Run detached:
#   tmux new-session -d -s msc_watcher \
#     'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
#      bash scripts/experiments/setup/multitask_ssl_corpora/watch_and_eval.sh \
#        > /tmp/msc_watcher.log 2>&1'
#
# Waits until, for EVERY one of the 8 runs (msc_{cov,all8}_{NM,CL,FP,MIX}):
#   (a) the newest state dir holds checkpoint/state_dict_30000.ckpt (terminal ckpt
#       — the trainer's off-by-one means no 40k ckpt is ever written), AND
#   (b) its training tmux session has exited (training truly finished => GPU free).
# Then: build model_list.txt (30k ckpts, keyed cov_NM..all8_MIX), run the frozen-
# encoder eval sweep on GPUs 0-3, and commit+push the refreshed result CSVs from
# this worktree onto the current branch. Ends the log with "ALL COMPLETE".
# On MAX_WAIT (default 8h) with runs still missing: logs the stragglers and exits 1
# with "WATCHER TIMEOUT — INCOMPLETE" (no eval, no commit).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"   # setup/<name> is 4 levels below repo root
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/state}"
POLL="${POLL:-300}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-28800}"
GPUS="${GPUS:-0,1,2,3}"
RUNS="cov_NM cov_CL cov_FP cov_MIX all8_NM all8_CL all8_FP all8_MIX"

say(){ echo "[msc_watcher $(date '+%F %H:%M:%S')] $*"; }

has_30k(){  # RUN -> 0 iff newest msc_<RUN>_<ts> state dir has the 30k ckpt
  local d
  d="$(ls -dt "${STATE_DIR}/msc_$1_"[0-9]*/ 2>/dev/null | head -n1 || true)"
  [[ -n "${d}" && -f "${d}checkpoint/state_dict_30000.ckpt" ]]
}
session_alive(){ tmux has-session -t "msc_$1" 2>/dev/null; }

start=$(date +%s)
while :; do
  pending=""
  for run in ${RUNS}; do
    if ! has_30k "${run}"; then pending+=" ${run}(no-30k)";
    elif session_alive "${run}"; then pending+=" ${run}(running)"; fi
  done
  if [[ -z "${pending}" ]]; then say "all 8 runs complete with 30k ckpts"; break; fi
  say "waiting on:${pending}"
  if (( $(date +%s) - start >= MAX_WAIT_SECS )); then
    say "WATCHER TIMEOUT — INCOMPLETE:${pending}"
    exit 1
  fi
  sleep "${POLL}"
done

say "building model list"
STATE_DIR="${STATE_DIR}" bash "${SCRIPT_DIR}/make_model_list.sh" || { say "model list FAILED"; exit 1; }

say "running eval sweep on GPUs ${GPUS}"
MODEL_LIST="${SCRIPT_DIR}/model_list.txt" \
  bash "${SCRIPT_DIR}/run_eval_sweep.sh" --gpus "${GPUS}" || { say "eval sweep FAILED"; exit 1; }

say "committing result CSVs"
cd "${REPO_ROOT}"
branch="$(git branch --show-current)"
say "branch=${branch} identity=$(git config user.name)/$(git config user.email)"
git add scripts/experiments/analysis/evaluation/shared_task_tables/*/data/*.csv 2>/dev/null
if git diff --cached --quiet; then
  say "no CSV changes to commit (parser may have written nothing new)"
else
  git commit -m "multitask_ssl_corpora: eval-sweep result CSVs (8 arms, 30k ckpts)" \
    -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>" \
    && git push origin "${branch}" \
    || { say "commit/push FAILED — CSVs remain in the worktree"; exit 1; }
fi

say "ALL COMPLETE"
