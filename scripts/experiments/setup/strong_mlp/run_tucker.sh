#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
GPU="${GPU:-3}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/state/strong_mlp/$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
TARGETS="${TARGETS:-covid_political,election2020,ukr_rus_suspended,twibot20}"
BUDGETS="${BUDGETS:-10,100,500,1000,-1}"
SEEDS="${SEEDS:-0,1,2,3,4}"

[[ "$GPU" =~ ^[0-3]$ ]] || { echo "refusing non-owned Tucker GPU $GPU" >&2; exit 2; }
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
export WANDB_DIR="$OUTPUT/wandb"
cd "$REPO_ROOT"

cmd=("${CONDA_PREFIX}/bin/python" -u -m scripts.experiments.setup.strong_mlp.train
  --output "$OUTPUT" --device cuda:0 --targets "$TARGETS" --budgets "$BUDGETS" --seeds "$SEEDS")
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q' "$GPU"; printf ' %q' "${cmd[@]}"; printf '\n'
  exit 0
fi

used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
(( used <= 2500 )) || { echo "GPU $GPU already uses ${used} MiB; refusing to collide" >&2; exit 3; }
mkdir -p "$(dirname "$OUTPUT")"
CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}" 2>&1 | tee "${OUTPUT}.log"
