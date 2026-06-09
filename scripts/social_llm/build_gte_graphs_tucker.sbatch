#!/bin/bash
#SBATCH --job-name=social-llm-gte-graphs
#SBATCH --output=/home1/eibl/gfm/prodigy/logs/%x_%j.out
#SBATCH --error=/home1/eibl/gfm/prodigy/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00

set -euo pipefail

RAW_ROOT="/dataMeR2/phil/data/social_llm_data"
OUT_ROOT="/dataMeR2/phil/data"
MODEL="Alibaba-NLP/gte-multilingual-base"
REVISION="9bbca17d9273fd0d03d5725c7a4b0f6b45142062"
EMB_NAME="user_bio_embeddings_gte_multilingual_base.pt"

module purge || true
source "$(conda info --base)/etc/profile.d/conda.sh"

cd /home1/eibl/gfm/prodigy
mkdir -p /home1/eibl/gfm/prodigy/logs

DATASETS=(
  "covid:covid_political"
  "covid_mf:covid_mf"
  "election2020:election2020"
  "hate_bots05:hate_bots05"
  "hate_bots08:hate_bots08"
  "ukr_rus_hate:ukr_rus_hate"
  "ukr_rus_suspended:ukr_rus_suspended"
)

for item in "${DATASETS[@]}"; do
  src_name="${item%%:*}"
  out_name="${item##*:}"
  src_dir="${RAW_ROOT}/${src_name}"
  out_dir="${OUT_ROOT}/${out_name}"
  emb_path="${out_dir}/embeddings/${EMB_NAME}"
  bio_root="${out_dir}/bio_embeddings/gte-multilingual-base/version=v001"

  echo "=========================================="
  echo "Building ${src_name} -> ${out_name}"
  echo "=========================================="

  mkdir -p "${out_dir}/embeddings" "${out_dir}/graphs" "${bio_root}"

  conda activate bio-embeddings-v001
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  python -u scripts/social_llm/build_bio_embeddings.py \
    --csv "${src_dir}/user_data.csv" \
    --out "${emb_path}" \
    --bio-output-root "${bio_root}" \
    --model "${MODEL}" \
    --revision "${REVISION}" \
    --batch-size 1024

  conda activate prodigy
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  python -u scripts/social_llm/generate_graph.py \
    --graph "${src_dir}/graph.pickle" \
    --csv "${src_dir}/user_data.csv" \
    --embeddings "${emb_path}" \
    --embedding_pool meanpool \
    --all_label_cols \
    --out_dir "${out_dir}/graphs" \
    --out "${out_dir}/graphs/retweet_graph.pt" \
    --write_default_copy
done

echo "Done building social_llm GTE graph artifacts."
