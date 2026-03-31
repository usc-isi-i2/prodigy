#!/bin/bash
#SBATCH --job-name=ukr-retweet
#SBATCH --output=/scratch1/eibl/data/ukr_rus_twitter/logs/ukr-retweet-%j.out
#SBATCH --error=/scratch1/eibl/data/ukr_rus_twitter/logs/ukr-retweet-%j.err
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# set -euo pipefail

module purge || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

cd /home1/eibl/gfm/prodigy

mkdir -p /scratch1/eibl/data/ukr_rus_twitter/logs
mkdir -p /scratch1/eibl/data/ukr_rus_twitter/graphs

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Load your Python environment here if needed.
# module load python/3.10
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate <env-name>

python data/data/ukr_rus_twitter/scripts/generate_retweet_graph.py \
  --csv "/project2/ll_774_951/uk_ru/twitter/data/*/*.csv" \
  --out /scratch1/eibl/data/ukr_rus_twitter/graphs/retweet_graph_100files.pt \
  --max_files 100
