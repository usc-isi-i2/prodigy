#!/usr/bin/env bash
set -euo pipefail
phase="${1:?prepare/select/freeze/refit}"
seed="${2:-0}"
gpu="${3:-0}"
[[ "$phase" =~ ^(prepare|select|freeze|refit)$ && "$seed" =~ ^[0-2]$ && "$gpu" =~ ^[0-3]$ ]]
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source /home/mhchu/miniconda3/etc/profile.d/conda.sh
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
python scripts/experiments/setup/ogbl_collab_compact_joint/candidate.py "$phase" \
 --runtime /dataMeR1/phil/gfm/ogbl_collab_compact_joint/joint_v1 \
 --out /dataMeR1/phil/gfm/ogbl_collab_compact_joint/candidate_fresh_v1 \
 --seed "$seed" --device "cuda:$gpu"
