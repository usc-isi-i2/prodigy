#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source /home/mhchu/miniconda3/etc/profile.d/conda.sh
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
cd /dataMeR1/phil/gfm/prodigy-collab-ncn-completion
entry=scripts/experiments/setup/ogbl_collab_ncn_completion
base=/dataMeR1/phil/gfm/ogbl_collab_compact_joint
runtime="$base/ncn_completion_v1"
common=(--source "$base/candidate_fresh_v1" --original "$base/joint_v1" --cache "$runtime/cache")
if [[ "$1" == prepare ]]; then
  python "$entry/test_core.py"
  python "$entry/prepare.py" --source "$base/candidate_fresh_v1" --original "$base/joint_v1" --out "$runtime/cache"
  for arm in control observed completion; do
    python "$entry/run.py" "${common[@]}" --out "$runtime/smoke" --arm "$arm" --seed 0 --device cuda:1 --smoke
  done
elif [[ "$1" == production ]]; then
  python "$entry/run.py" "${common[@]}" --out "$runtime/runs" --dry-run
  for arm in control observed completion; do
    for seed in 0 1 2; do
      python "$entry/run.py" "${common[@]}" --out "$runtime/runs" --arm "$arm" --seed "$seed" --device cuda:1
    done
  done
else
  exit 2
fi
