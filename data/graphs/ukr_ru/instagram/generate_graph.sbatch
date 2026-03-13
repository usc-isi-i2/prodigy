#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --job-name=co_hashtag
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

export LD_LIBRARY_PATH=/home1/eibl/.conda/envs/prodigy/lib:$LD_LIBRARY_PATH
conda run -n prodigy python /home1/eibl/gfm/prodigy/data/graphs/ukr_ru/instagram/generate_graph_instagram_co_hashtag.py
