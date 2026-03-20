#!/bin/bash
# Sweep: 3 pretrained models × 4 shots × 3 ways for overperformer classification
# Usage: bash scripts/sweep_instagram_overperformer.sh

MODELS=(
    "pretrain_instagram_nm_11_03_2026_15_42_26"
    "pretrain_instagram_nm_11_03_2026_14_44_15"
    "pretrain_instagram_nm_11_03_2026_14_09_51"
)
SHOTS=(1 3 5 10)
WAYS=(2 3 10)

for MODEL in "${MODELS[@]}"; do
    # Find latest checkpoint for this run
    CKPT=$(ls /home1/eibl/gfm/prodigy/state/${MODEL}/checkpoint/state_dict_*.ckpt 2>/dev/null | sort -V | tail -1)
    if [ -z "$CKPT" ]; then
        echo "⚠ No checkpoint found for $MODEL, skipping"
        continue
    fi
    echo "Using checkpoint: $CKPT"

    for SHOT in "${SHOTS[@]}"; do
        for WAY in "${WAYS[@]}"; do
            PREFIX="eval_overperformer_${MODEL}_${SHOT}shot_${WAY}way"
            echo "Submitting: $PREFIX"
            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ig_overperf
#SBATCH --output=logs/${PREFIX}_%j.out
#SBATCH --error=logs/${PREFIX}_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=00:30:00

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate prodigy
export LD_PRELOAD=\$CONDA_PREFIX/lib/libstdc++.so.6

cd /home1/eibl/gfm/prodigy
mkdir -p logs

python experiments/run_single_experiment.py \\
    --dataset instagram_mention \\
    --root data/graphs/ukr_ru/instagram \\
    --graph_filename mention_graph_overperformer_bge.pt \\
    --input_dim 1024 \\
    --original_features True \\
    --task classification \\
    --device 0 \\
    -val_cap 500 \\
    -test_cap 500 \\
    --workers 8 \\
    -shot ${SHOT} \\
    -way ${WAY} \\
    --pretrained_model_run ${CKPT} \\
    --prefix ${PREFIX}
EOF
        done
    done
done
