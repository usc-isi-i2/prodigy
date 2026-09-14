#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?usage: launch_tucker.sh GPU SEED}"
SEED="${2:?usage: launch_tucker.sh GPU SEED}"
case "$GPU" in 0|1|2|3) ;; *) exit 2 ;; esac
case "$SEED" in 0|1|2) ;; *) exit 2 ;; esac
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source /home/mhchu/miniconda3/etc/profile.d/conda.sh
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /dataMeR1/phil/gfm/prodigy-collab-joint
RUN_ROOT="${RUN_ROOT:-/dataMeR1/phil/gfm/ogbl_collab_compact_joint/joint_v1}"
for ARM in structure joint; do
  python -u scripts/experiments/setup/ogbl_collab_compact_joint/run.py train \
    --out "$RUN_ROOT" --arm "$ARM" --seed "$SEED" --device "cuda:$GPU"
done
