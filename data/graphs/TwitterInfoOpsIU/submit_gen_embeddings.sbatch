#!/bin/bash
#SBATCH --job-name=generate_embeddings
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00


source $(conda info --base)/etc/profile.d/conda.sh
conda activate prodigy

cd "/home1/eibl/gfm/prodigy/data/graphs/TwitterInfoOpsIU"

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python generate_embeddings.py
