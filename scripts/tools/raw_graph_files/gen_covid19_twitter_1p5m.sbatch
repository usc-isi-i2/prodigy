#!/bin/bash
#SBATCH --job-name=gen_covid19_1p5m
#SBATCH --output=/home1/eibl/gfm/prodigy/logs/%x_%j.out
#SBATCH --error=/home1/eibl/gfm/prodigy/logs/%x_%j.err
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

module purge || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

cd /home1/eibl/gfm/prodigy
mkdir -p /home1/eibl/gfm/prodigy/logs /scratch1/eibl/data/covid19_twitter/graphs

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -u data/data/covid19_twitter/scripts/generate_user_graph.py \
  --json_glob "/scratch1/eibl/data/covid19_twitter/raw/*/*.json" \
  --max_nodes 1500000 \
  --max_files 0 \
  --history_fraction 0.3 \
  --embeddings /scratch1/eibl/data/covid19_twitter/embeddings/user_embeddings_minilm.pt \
  --embedding_pool meanpool \
  --out /scratch1/eibl/data/covid19_twitter/graphs/retweet_graph_1p5m_hf03_labeled.pt
