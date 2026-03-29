#!/bin/bash
#SBATCH --job-name=ukr-retweet
#SBATCH --output=logs/ukr-retweet-%j.out
#SBATCH --error=logs/ukr-retweet-%j.err
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../../../" && pwd)"
cd "$ROOT_DIR"

mkdir -p logs

# Load your Python environment here if needed.
# module load python/3.10
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate <env-name>

python data/data/ukr_rus_twitter/scripts/generate_retweet_graph.py \
  --csv "/project2/ll_774_951/uk_ru/twitter/data/*/*.csv" \
  --out data/data/ukr_rus_twitter/graphs/retweet_graph.pt \
  --max_files 200
