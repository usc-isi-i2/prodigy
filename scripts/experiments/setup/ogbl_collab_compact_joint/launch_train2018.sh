#!/usr/bin/env bash
set -euo pipefail
seed="${1:?seed0-2 required}"
gpu="${2:?owned GPU0-3 required}"
[[ "$seed" =~ ^[0-2]$ && "$gpu" =~ ^[0-3]$ ]]
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source /home/mhchu/miniconda3/etc/profile.d/conda.sh
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python scripts/experiments/setup/ogbl_collab_compact_joint/train2018.py \
  --runtime /dataMeR1/phil/gfm/ogbl_collab_compact_joint/joint_v1 \
  --test-runtime /dataMeR1/phil/gfm/ogbl_collab_compact_joint/official_test_fusion_v1 \
  --out /dataMeR1/phil/gfm/ogbl_collab_compact_joint/train2018_v1 \
  --seed "$seed" --device "cuda:$gpu"
