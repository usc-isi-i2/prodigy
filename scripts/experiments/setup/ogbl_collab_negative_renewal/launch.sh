#!/bin/bash
set -euo pipefail
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
cd /dataMeR1/phil/gfm/prodigy-negative-renewal
python scripts/experiments/setup/ogbl_collab_negative_renewal/run.py dry-run
python scripts/experiments/setup/ogbl_collab_negative_renewal/run.py prepare
for seed in 0 1 2; do
  for arm in fixed renew; do
    python scripts/experiments/setup/ogbl_collab_negative_renewal/run.py train --seed "$seed" --arm "$arm"
  done
done
