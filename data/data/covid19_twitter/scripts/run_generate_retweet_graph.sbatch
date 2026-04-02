#!/bin/bash
#SBATCH --job-name=covid19-retweet
#SBATCH --output=/scratch1/eibl/data/covid19_twitter/logs/covid19-retweet-%j.out
#SBATCH --error=/scratch1/eibl/data/covid19_twitter/logs/covid19-retweet-%j.err
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

module purge || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

cd /home1/eibl/gfm/prodigy
mkdir -p /scratch1/eibl/data/covid19_twitter/logs
mkdir -p /scratch1/eibl/data/covid19_twitter/graphs
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -u data/data/covid19_twitter/scripts/generate_retweet_graph.py \
  --json_glob "/scratch1/eibl/data/covid19_twitter/*/*.json" \
  --out /scratch1/eibl/data/covid19_twitter/graphs/retweet_graph_minilm.pt \
  --embeddings /scratch1/eibl/data/covid19_twitter/embeddings/user_embeddings_minilm.pt \
  --embedding_pool meanpool
