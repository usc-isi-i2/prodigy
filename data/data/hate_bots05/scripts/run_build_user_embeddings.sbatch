#!/bin/bash
#SBATCH --job-name=hate_bots05-emb
#SBATCH --output=/scratch1/eibl/logs/hate_bots05/emb-%j.out
#SBATCH --error=/scratch1/eibl/logs/hate_bots05/emb-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00

module purge || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
cd /home1/eibl/gfm/prodigy
mkdir -p /scratch1/eibl/logs/hate_bots05
mkdir -p /scratch1/eibl/data/hate_bots05/embeddings
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python -u data/data/covid_political/scripts/build_user_embeddings.py \
  --csv /project2/emiliofe_74/julie/social_llm_data/hate_bots05/user_data.csv \
  --out /scratch1/eibl/data/hate_bots05/embeddings/user_embeddings_minilm.pt
