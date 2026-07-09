#!/usr/bin/env bash
# Overnight self-orchestration: wait for the B0/B1/E1 pretrains to reach their final
# checkpoint (or a time cap), then run the full downstream (eval + diagnostics +
# parse). Launch detached in tmux so it survives disconnects. Robust to arms that
# never finish: after MAX_WAIT it proceeds on whatever checkpoints exist.
set -uo pipefail
DIR=scripts/experiments/topology_feature_ssl
STATE_DIR="${STATE_DIR:-/dataMeR1/phil/gfm/prodigy/state}"
TARGET_STEP="${TARGET_STEP:-120000}"
MAX_WAIT_SECS="${MAX_WAIT_SECS:-25200}"   # 7h, then proceed on latest checkpoints
POLL="${POLL:-300}"
say(){ echo "[orchestrate $(date +%H:%M:%S)] $*"; }

have_final(){  # arm -> success if a checkpoint step >= TARGET_STEP exists
  local d; d=$(ls -dt "${STATE_DIR}/tfssl_$1_"*/ 2>/dev/null | head -1)
  [ -z "$d" ] && return 1
  local step
  step=$(ls "${d}checkpoint/"state_dict_*.ckpt 2>/dev/null \
    | sed -E 's#.*state_dict_([0-9]+)\.ckpt#\1#' | sort -n | tail -1)
  [ -z "$step" ] && return 1
  [ "$step" -ge "$TARGET_STEP" ]
}

start=$(date +%s)
while :; do
  ok=0; for a in B0 B1 E1; do have_final "$a" && ok=$((ok+1)); done
  say "arms at final ($TARGET_STEP): $ok/3"
  [ "$ok" -ge 3 ] && { say "all finals ready"; break; }
  [ $(( $(date +%s) - start )) -ge "$MAX_WAIT_SECS" ] && { say "MAX_WAIT reached; proceeding on latest"; break; }
  sleep "$POLL"
done

say "launching downstream"
GPUS="${GPUS:-0,1,2,3}" STATE_DIR="$STATE_DIR" bash "$DIR/run_downstream_tucker.sh"
say "ORCHESTRATE_DONE"
