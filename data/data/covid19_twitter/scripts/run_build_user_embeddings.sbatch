#!/bin/bash
#SBATCH --job-name=covid19-twitter-emb
#SBATCH --output=/scratch1/eibl/data/covid19_twitter/logs/covid19-twitter-emb-%j.out
#SBATCH --error=/scratch1/eibl/data/covid19_twitter/logs/covid19-twitter-emb-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

module purge || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

cd /home1/eibl/gfm/prodigy
mkdir -p /scratch1/eibl/data/covid19_twitter/logs
mkdir -p /scratch1/eibl/data/covid19_twitter/embeddings
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -u data/data/covid19_twitter/scripts/build_user_embeddings.py \
  --json_glob "/scratch1/eibl/data/covid19_twitter/*/*.json" \
  --out /scratch1/eibl/data/covid19_twitter/embeddings/user_embeddings_minilm.pt \
  --batch_size 32 \
  --checkpoint_path /scratch1/eibl/data/covid19_twitter/embeddings/user_embeddings_minilm.checkpoint.pkl \
  --checkpoint_every 5 \
  --resume
