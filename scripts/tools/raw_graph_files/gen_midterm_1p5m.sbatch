#!/bin/bash
#SBATCH --job-name=gen_midterm_1p5m
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
mkdir -p /home1/eibl/gfm/prodigy/logs /scratch1/eibl/data/midterm/graphs

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -u data/data/midterm/scripts/generate_user_graph.py \
  --csv_glob "/project2/ll_774_951/midterm/*/*.csv" \
  --max_nodes 1500000 \
  --max_files 0 \
  --history_fraction 0.3 \
  --embeddings /scratch1/eibl/data/midterm/embeddings/embeddings_all-MiniLM-L6-v2.pt \
  --embedding_pool meanpool \
  --out /scratch1/eibl/data/midterm/graphs/retweet_graph_1p5m.pt
