#!/bin/bash
#SBATCH --job-name=gen_ukr_rus_1p5m
#SBATCH --output=/home1/eibl/gfm/prodigy/logs/%x_%j.out
#SBATCH --error=/home1/eibl/gfm/prodigy/logs/%x_%j.err
#SBATCH --partition=main
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

module purge || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

cd /home1/eibl/gfm/prodigy
mkdir -p /home1/eibl/gfm/prodigy/logs /scratch1/eibl/data/ukr_rus_twitter/graphs

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -u data/data/ukr_rus_twitter/scripts/generate_retweet_graph.py \
  --csv "/project2/ll_774_951/uk_ru/twitter/data/*/*.csv" \
  --max_nodes 1500000 \
  --max_files 0 \
  --history_fraction 0.3 \
  --embeddings /scratch1/eibl/data/ukr_rus_twitter/embeddings/user_embeddings_minilm.pt \
  --embedding_pool meanpool \
  --pseudo-political-labels \
  --out /scratch1/eibl/data/ukr_rus_twitter/graphs/retweet_graph_1p5m_hf03_political_labels.pt
