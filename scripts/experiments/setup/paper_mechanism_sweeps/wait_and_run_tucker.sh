#!/usr/bin/env bash
# Serialize after the already queued paper campaigns, then require a stable idle window.
set -euo pipefail

HEAVY_SESSIONS=(
  paper-optimized-queue paper-onehop-seed2-overlap paper-optimized-relauncher
  vision-mixture-seeds paper-core-fast-eval paper-all-pairs
)
for session in "${HEAVY_SESSIONS[@]}"; do
  while tmux has-session -t "$session" 2>/dev/null; do sleep 30; done
done
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
export PATH="/home/mhchu/miniconda3/bin:$PATH"
bash "$(dirname "$0")/run_tucker.sh"
