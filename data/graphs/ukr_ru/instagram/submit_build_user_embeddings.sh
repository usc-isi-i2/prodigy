#!/bin/bash
#SBATCH --job-name=ig_user_embeddings
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate prodigy

cd /home1/eibl/gfm/prodigy/data/graphs/ukr_ru/instagram

mkdir -p logs

python build_user_embeddings.py
