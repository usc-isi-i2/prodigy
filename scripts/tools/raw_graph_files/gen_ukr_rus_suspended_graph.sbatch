#!/bin/bash
#SBATCH --job-name=gen_ukr_rus_suspended_1p5m
#SBATCH --output=/home1/eibl/gfm/prodigy/logs/%x_%j.out
#SBATCH --error=/home1/eibl/gfm/prodigy/logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail

module purge || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

cd /home1/eibl/gfm/prodigy
mkdir -p /home1/eibl/gfm/prodigy/logs /scratch1/eibl/data/ukr_rus_suspended/graphs

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -u data/data/social_llm/scripts/generate_graph.py \
  --graph /project2/emiliofe_74/julie/social_llm_data/ukr_rus_suspended/graph.pickle \
  --csv /project2/emiliofe_74/julie/social_llm_data/ukr_rus_suspended/user_data.csv \
  --embeddings /scratch1/eibl/data/ukr_rus_suspended/embeddings/user_embeddings_minilm.pt \
  --label_col label_suspended \
  --out /scratch1/eibl/data/ukr_rus_suspended/graphs/retweet_graph.pt
