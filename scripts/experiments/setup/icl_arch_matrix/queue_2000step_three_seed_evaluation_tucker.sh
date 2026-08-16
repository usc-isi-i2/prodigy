#!/usr/bin/env bash
set -euo pipefail

while tmux has-session -t archsat2000 2>/dev/null; do
  echo "waiting for archsat2000 training utc=$(date -u +%FT%TZ)"
  sleep 60
done

cd /dataMeR1/phil/gfm/prodigy-archsat-eval
exec bash scripts/experiments/setup/icl_arch_matrix/run_2000step_three_seed_evaluation_tucker.sh
