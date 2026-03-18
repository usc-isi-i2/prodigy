#!/bin/bash
#SBATCH --job-name=gen_retweet_ukr_rus
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00

# set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

cd /home1/eibl/gfm/prodigy/data/graphs/midterm
mkdir -p logs
mkdir -p /home1/eibl/gfm/prodigy/data/graphs/ukr_rus_twitter

python -u generate_retweet_graph.py --max_files 25 --csv "/project2/ll_774_951/uk_ru/twitter/data/*/*.csv" --out "/home1/eibl/gfm/prodigy/data/graphs/ukr_ru/twitter/retweet_graph.pt"
