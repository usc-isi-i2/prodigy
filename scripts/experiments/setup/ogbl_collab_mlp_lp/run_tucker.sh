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
if [[ "$branch" != "codex/ogbl-collab-mlp" ]]; then
  echo "expected codex/ogbl-collab-mlp, found $branch" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "refusing to launch from a dirty Tucker worktree" >&2
  exit 1
fi

run_root="/dataMeR1/phil/gfm/ogbl_collab_mlp_lp/official_v1"
mkdir -p "$run_root/logs"
log="$run_root/logs/gpu${device}_seeds_$(IFS=_; echo "${seeds[*]}").log"
exec > >(tee -a "$log") 2>&1

echo "revision=$(git rev-parse HEAD) branch=$branch device=$device seeds=${seeds[*]}"
for seed in "${seeds[@]}"; do
  out="$run_root/seed${seed}"
  if [[ -f "$out/results.json" ]]; then
    if python -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("complete") else 1)' "$out/results.json"; then
      echo "seed=$seed already complete; skipping"
      continue
    fi
  fi
  if [[ -e "$out" ]]; then
    echo "seed=$seed has an incomplete existing output at $out; refusing to overwrite" >&2
    exit 1
  fi
  python scripts/experiments/setup/ogbl_collab_mlp_lp/run.py \
    --dataset-root /dataMeR1/phil/data/ogb \
    --out "$out" \
    --seed "$seed" \
    --device "cuda:${device}" \
    --epochs 400 \
    --patience 20 \
    --val-interval 1 \
    --batch-size 65536 \
    --eval-batch-size 262144 \
    --learning-rate 0.001 \
    --weight-decay 0.0005 \
    --grad-clip-norm 1.0 \
    --wandb-mode online \
    --wandb-project ogbl-collab-mlp-lp \
    --run-tag official_v1
done
