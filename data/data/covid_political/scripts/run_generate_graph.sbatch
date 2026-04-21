#!/bin/bash
#SBATCH --job-name=covid-pol-graph
#SBATCH --output=/scratch1/eibl/data/social_llm_covid/logs/graph-%j.out
#SBATCH --error=/scratch1/eibl/data/social_llm_covid/logs/graph-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

module purge || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

cd /home1/eibl/gfm/prodigy

mkdir -p /scratch1/eibl/data/social_llm_covid/logs
mkdir -p /scratch1/eibl/data/covid_political/graphs

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -u data/data/covid_political/scripts/generate_graph.py \
  --graph /project2/emiliofe_74/julie/social_llm_data/covid/graph.pickle \
  --csv /project2/emiliofe_74/julie/social_llm_data/covid/user_data.csv \
  --embeddings /scratch1/eibl/data/covid_political/embeddings/user_embeddings_minilm.pt \
  --out /scratch1/eibl/data/covid_political/graphs/retweet_graph.pt
