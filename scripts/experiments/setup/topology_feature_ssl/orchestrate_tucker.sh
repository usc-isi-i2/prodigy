#!/usr/bin/env bash
# Overnight self-orchestration: wait for the B0/B1/E1 pretrain tmux sessions to END
# (finished => GPUs freed), then run the full downstream (eval + diagnostics +
# parse) on all GPUs. Waiting on session-exit (not a step count) guarantees the
# eval jobs don't contend with training for the GPUs. After MAX_WAIT it kills any
# lingering pretrains to free the GPUs and proceeds on whatever checkpoints exist.
set -uo pipefail
DIR=scripts/experiments/topology_feature_ssl
MAX_WAIT_SECS="${MAX_WAIT_SECS:-21600}"   # 6h, then kill lingering pretrains + proceed
POLL="${POLL:-300}"
say(){ echo "[orchestrate $(date +%H:%M:%S)] $*"; }

sessions_running(){ tmux ls 2>/dev/null | grep -cE "^tfssl_(B0|B1|E1):"; }

start=$(date +%s)
while :; do
  n=$(sessions_running)
  say "pretrain sessions still running: $n/3"
  [ "$n" -eq 0 ] && { say "all pretrains finished (GPUs free)"; break; }
  if [ $(( $(date +%s) - start )) -ge "$MAX_WAIT_SECS" ]; then
    say "MAX_WAIT reached — killing lingering pretrains to free GPUs"
    for a in B0 B1 E1; do tmux kill-session -t "tfssl_$a" 2>/dev/null; done
    sleep 30; break
  fi
  sleep "$POLL"
done

say "launching downstream"
GPUS="${GPUS:-0,1,2,3}" bash "$DIR/run_downstream_tucker.sh"
say "ORCHESTRATE_DONE"
