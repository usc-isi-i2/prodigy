#!/usr/bin/env bash
# Seed-ordered, restart-safe RQ1 queue: seed 0 completely, then 1, then 2.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/rq1_label_efficiency_loto}"
mkdir -p "$LOG_ROOT"
for seed in 0 1 2; do
  echo "SEED $seed PRETRAIN START $(date -u +%FT%TZ)" | tee -a "$LOG_ROOT/progress.log"
  SEED="$seed" bash "$SCRIPT_DIR/run_pretrain_seed_tucker.sh"
  echo "SEED $seed ADAPT START $(date -u +%FT%TZ)" | tee -a "$LOG_ROOT/progress.log"
  SEED="$seed" bash "$SCRIPT_DIR/run_adapt_seed_tucker.sh"
  echo "SEED $seed COMPLETE $(date -u +%FT%TZ)" | tee -a "$LOG_ROOT/progress.log"
done
