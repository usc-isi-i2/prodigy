#!/usr/bin/env bash
set -euo pipefail

while tmux has-session -t archsat2000-eval 2>/dev/null; do
  echo "waiting for archsat2000-eval utc=$(date -u +%FT%TZ)"
  sleep 60
done

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
cd /dataMeR1/phil/gfm/prodigy-archsat-final
exec python -u scripts/experiments/analysis/transfer/matrices/cross_architecture/icl_arch_matrix/aggregate_arch_saturation_2000.py \
  --results-root /dataMeR1/phil/gfm/prodigy-archsat-eval/log/icl_arch_saturation_2000_eval/results \
  --output-root /dataMeR1/phil/gfm/prodigy-archsat-final/output/icl_arch_saturation_2000
