#!/bin/bash
#SBATCH --job-name=ig_user_embeddings
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate prodigy

cd "$(dirname "$0")"

mkdir -p logs

python build_user_embeddings.py
