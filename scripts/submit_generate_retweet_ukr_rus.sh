#!/bin/bash
#SBATCH --job-name=gen_retweet_ukr_rus
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

cd /home1/eibl/gfm/prodigy/data/graphs/midterm
mkdir -p logs
mkdir -p /home1/eibl/gfm/prodigy/data/graphs/ukr_rus_twitter

python generate_retweet_graph.py --csv "/project2/ll_774_951/ukr_rus_twitter/*/*.csv" --out "/home1/eibl/gfm/prodigy/data/graphs/ukr_rus_twitter/retweet_graph.pt" --max_files 500
