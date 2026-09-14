#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source /home/mhchu/miniconda3/etc/profile.d/conda.sh
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
cd /dataMeR1/phil/gfm/prodigy-collab-warm-joint
entry=scripts/experiments/setup/ogbl_collab_warm_joint/run.py
runtime=/dataMeR1/phil/gfm/ogbl_collab_compact_joint
python "$entry" --out "$runtime/warm_joint_v1" --source "$runtime/candidate_fresh_v1" --original "$runtime/joint_v1" --dry-run
for arm in bce warm; do
  for seed in 0 1 2; do
    python "$entry" --out "$runtime/warm_joint_v1" --source "$runtime/candidate_fresh_v1" --original "$runtime/joint_v1" --arm "$arm" --seed "$seed" --device cuda:1
  done
done
