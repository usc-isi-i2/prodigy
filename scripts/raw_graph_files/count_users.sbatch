#!/bin/bash
#SBATCH --job-name=count_users
#SBATCH --output=/home1/eibl/gfm/prodigy/logs/%x_%j.out
#SBATCH --error=/home1/eibl/gfm/prodigy/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

SCRIPT_PATH="${1:-scripts/raw_graph_files/count_users_ukr_rus.py}"

module purge || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

cd /home1/eibl/gfm/prodigy
mkdir -p /home1/eibl/gfm/prodigy/logs

python -u "$SCRIPT_PATH"
