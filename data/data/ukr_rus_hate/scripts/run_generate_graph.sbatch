#!/bin/bash
#SBATCH --job-name=ukr_rus_hate-graph
#SBATCH --output=/scratch1/eibl/logs/ukr_rus_hate/graph-%j.out
#SBATCH --error=/scratch1/eibl/logs/ukr_rus_hate/graph-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

module purge || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
cd /home1/eibl/gfm/prodigy
mkdir -p /scratch1/eibl/logs/ukr_rus_hate
mkdir -p /scratch1/eibl/data/ukr_rus_hate/graphs
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -u data/data/social_llm/scripts/generate_graph.py \
  --graph /project2/emiliofe_74/julie/social_llm_data/ukr_rus_hate/graph.pickle \
  --csv   /project2/emiliofe_74/julie/social_llm_data/ukr_rus_hate/user_data.csv \
  --embeddings /scratch1/eibl/data/ukr_rus_hate/embeddings/user_embeddings_minilm.pt \
  --label_col label_toxicity \
  --out /scratch1/eibl/data/ukr_rus_hate/graphs/retweet_graph.pt
