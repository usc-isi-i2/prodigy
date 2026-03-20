#!/bin/bash
#SBATCH --job-name=ig_user_embeddings
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00

source $(conda info --base)/etc/profile.d/conda.sh
conda activate prodigy
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

cd /home1/eibl/gfm/prodigy/data/graphs/ukr_ru/instagram
mkdir -p logs

python build_user_embeddings.py


srun --partition=gpu --gres=gpu:1 --cpus-per-task=12 --mem=64G --time=02:00:00 --pty
bash -lc '
module purge
cd /home1/eibl/gfm/prodigy/data/graphs/midterm
conda run -n prodigy python -c "import sys, numpy; print(sys.executable);
print(numpy.__version__)"
conda run -n prodigy python build_user_embeddings.py \
--csv "/project2/ll_774_951/uk_ru/twitter/data/2022-02/*.csv" \
--out "/home1/eibl/gfm/prodigy/data/graphs/ukr_ru/twitter/user_embeddings_minilm.pt" \
--max-files 25
'