#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 DEVICE SEED [SEED ...]" >&2
  exit 2
fi
device="$1"
shift
seeds=("$@")
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
branch="$(git branch --show-current)"
[[ "$branch" == "codex/ogbl-collab-sage" ]] || { echo "unexpected branch $branch" >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "refusing dirty worktree" >&2; exit 1; }
run_root="/dataMeR1/phil/gfm/ogbl_collab_sage_lp/ogb_style_v1"
mkdir -p "$run_root/logs"
log="$run_root/logs/gpu${device}_seeds_$(IFS=_; echo "${seeds[*]}").log"
exec > >(tee -a "$log") 2>&1
echo "revision=$(git rev-parse HEAD) branch=$branch device=$device seeds=${seeds[*]}"
for seed in "${seeds[@]}"; do
  out="$run_root/seed${seed}"
  [[ ! -e "$out" ]] || { echo "refusing to overwrite $out" >&2; exit 1; }
  python scripts/experiments/setup/ogbl_collab_sage_lp/run_ogb_style.py \
    --dataset-root /dataMeR1/phil/data/ogb --out "$out" --seed "$seed" \
    --device "cuda:${device}" --epochs 400 --batch-size 65536 --eval-batch-size 262144 \
    --learning-rate 0.001 --wandb-mode online --wandb-project ogbl-collab-sage-lp \
    --run-tag ogb_style_v1
done
