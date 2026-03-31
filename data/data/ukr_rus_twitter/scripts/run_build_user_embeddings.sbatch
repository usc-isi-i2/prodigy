#!/bin/bash
#SBATCH --job-name=ukr-twitter-emb
#SBATCH --output=/scratch1/eibl/data/ukr_rus_twitter/logs/ukr-twitter-emb-%j.out
#SBATCH --error=/scratch1/eibl/data/ukr_rus_twitter/logs/ukr-twitter-emb-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

module purge || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

cd /home1/eibl/gfm/prodigy

mkdir -p /scratch1/eibl/data/ukr_rus_twitter/logs
mkdir -p /scratch1/eibl/data/ukr_rus_twitter/embeddings

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python data/data/ukr_rus_twitter/scripts/build_user_embeddings.py \
  --csv_glob "/project2/ll_774_951/uk_ru/twitter/data/*/*.csv" \
  --out /scratch1/eibl/data/ukr_rus_twitter/embeddings/user_embeddings_minilm.pt \
  --batch_size 32 \
  --checkpoint_path /scratch1/eibl/data/ukr_rus_twitter/embeddings/user_embeddings_minilm.checkpoint.pkl \
  --checkpoint_every 10 \
  --resume \
  --max_files 150
