#!/bin/bash
# ICL sweep: pretrained Instagram NM model (minilm) × shots × ways × labels
# Usage: bash scripts/sweep_instagram_icl_minilm.sh [model_run_name]
#   model_run_name: optional, defaults to latest pretrain_instagram_nm checkpoint

SHOTS=(1 3 5 10)
WAYS=(3)
LABELS=(overperformer)

# ── Resolve checkpoint ────────────────────────────────────────────────────────
if [ -n "$1" ]; then
    MODEL="$1"
else
    MODEL=$(ls -d /home1/eibl/gfm/prodigy/state/pretrain_instagram_nm_* 2>/dev/null \
            | sort -V | tail -1 | xargs basename)
    if [ -z "$MODEL" ]; then
        echo "No pretrain_instagram_nm checkpoint found. Run submit_pretrain_instagram_nm.sh first."
        exit 1
    fi
fi

CKPT=$(ls /home1/eibl/gfm/prodigy/state/${MODEL}/checkpoint/state_dict_*.ckpt 2>/dev/null | sort -V | tail -1)
if [ -z "$CKPT" ]; then
    echo "No checkpoint file found under state/${MODEL}/checkpoint/"
    exit 1
fi

echo "Model: $MODEL"
echo "Checkpoint: $CKPT"
echo "Labels: ${LABELS[*]}  |  Shots: ${SHOTS[*]}  |  Ways: ${WAYS[*]}"
echo "Total jobs: $(( ${#LABELS[@]} * ${#SHOTS[@]} * ${#WAYS[@]} ))"
echo ""

# ── Submit jobs ───────────────────────────────────────────────────────────────
for LABEL in "${LABELS[@]}"; do
    for SHOT in "${SHOTS[@]}"; do
        for WAY in "${WAYS[@]}"; do
            PREFIX="eval_${LABEL}_${MODEL}_${SHOT}shot_${WAY}way"
            echo "Submitting: $PREFIX"
            sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ig_icl_${LABEL}
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
    --graph_filename mention_graph_${LABEL}_minilm.pt \\
    --input_dim 393 \\
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
