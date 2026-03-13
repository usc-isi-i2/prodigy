#!/bin/bash
# Transfer sweep: Instagram-pretrained model evaluated on Twitter retweet graph
# Tests NM (structure transfer) and classification (few-shot political/follower labels)
#
# Usage: bash scripts/sweep_twitter_transfer_minilm.sh [model_run_name]
#   model_run_name: optional, defaults to latest pretrain_instagram_nm checkpoint

SHOTS=(1 3 5 10)
LABELS=(repdem)

# ── Resolve checkpoint ────────────────────────────────────────────────────────
if [ -n "$1" ]; then
    MODEL="$1"
else
    MODEL=$(ls -d /home1/eibl/gfm/prodigy/state/pretrain_instagram_nm_* 2>/dev/null \
            | sort -V | tail -1 | xargs basename)
    if [ -z "$MODEL" ]; then
        echo "No pretrain_instagram_nm checkpoint found."
        exit 1
    fi
fi

CKPT=$(ls /home1/eibl/gfm/prodigy/state/${MODEL}/checkpoint/state_dict_*.ckpt 2>/dev/null | sort -V | tail -1)
if [ -z "$CKPT" ]; then
    echo "No checkpoint file found under state/${MODEL}/checkpoint/"
    exit 1
fi

echo "Model:      $MODEL"
echo "Checkpoint: $CKPT"
echo ""

# ── 1. Neighbor matching (no labels — pure structure transfer) ────────────────
PREFIX_NM="eval_twitter_nm_${MODEL}"
echo "Submitting NM: $PREFIX_NM"
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tw_nm_transfer
#SBATCH --output=logs/${PREFIX_NM}_%j.out
#SBATCH --error=logs/${PREFIX_NM}_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate prodigy
export LD_PRELOAD=\$CONDA_PREFIX/lib/libstdc++.so.6

cd /home1/eibl/gfm/prodigy
mkdir -p logs

python experiments/run_single_experiment.py \\
    --dataset instagram_mention \\
    --root data/graphs/midterm \\
    --graph_filename retweet_graph_minilm.pt \\
    --input_dim 393 \\
    --original_features True \\
    --task neighbor_matching \\
    --device 0 \\
    -val_cap 100 \\
    -test_cap 100 \\
    --workers 4 \\
    -shot 5 \\
    -way 5 \\
    --pretrained_model_run ${CKPT} \\
    --prefix ${PREFIX_NM}
EOF

# ── 2. Classification sweep (political + follower tier, varying shots) ────────
echo ""
echo "Total classification jobs: $(( ${#LABELS[@]} * ${#SHOTS[@]} ))"
for LABEL in "${LABELS[@]}"; do
    for SHOT in "${SHOTS[@]}"; do
        PREFIX="eval_twitter_${LABEL}_${MODEL}_${SHOT}shot"
        echo "Submitting: $PREFIX"
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=tw_cls_${LABEL}
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
    --root data/graphs/midterm \\
    --graph_filename retweet_graph_${LABEL}_minilm.pt \\
    --input_dim 393 \\
    --original_features True \\
    --task classification \\
    --device 0 \\
    -val_cap 500 \\
    -test_cap 500 \\
    --workers 8 \\
    -shot ${SHOT} \\
    -way 3 \\
    --pretrained_model_run ${CKPT} \\
    --prefix ${PREFIX}
EOF
    done
done
