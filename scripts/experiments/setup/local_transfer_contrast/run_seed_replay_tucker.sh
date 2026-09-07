#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <checkpoint-seed:1|2> <stream:original|fresh> <owned-gpu:0..3>" >&2
  exit 2
fi

seed="$1"
stream="$2"
device="$3"
if [[ ! "$seed" =~ ^[12]$ ]] || [[ ! "$stream" =~ ^(original|fresh)$ ]] || [[ ! "$device" =~ ^[0-3]$ ]]; then
  echo "invalid seed, stream, or device" >&2
  exit 2
fi

runtime_root="log/local_transfer_contrast/seed_replication"
model_list="${runtime_root}/models_seed${seed}.tsv"
output="${runtime_root}/seed${seed}_${stream}"
offset=0
if [[ "$stream" == "fresh" ]]; then
  offset=100003
fi
if [[ ! -f "$model_list" ]]; then
  echo "missing $model_list" >&2
  exit 1
fi
if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 1
fi

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline

python -m scripts.experiments.setup.target_performance_mechanisms.replay \
  --model-list "$model_list" \
  --output "$output" \
  --datasets covid_political,facebook_page_reference,election2020,twibot20,ukr_rus_suspended \
  --specialists-only \
  --variants baseline \
  --device "$device" \
  --batch-count 32 \
  --threads 8 \
  --save-embeddings \
  --training-seed 0 \
  --eval-episode-seed-offset "$offset" \
  --trace-parity-atol 1e-5
