#!/bin/bash
#SBATCH --job-name=midterm_pl_icl
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-5

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

cd "$(dirname "$0")/.."
mkdir -p logs

SHOTS=(0 1 3 5 10 20)
SHOT="${SHOTS[$SLURM_ARRAY_TASK_ID]}"

GRAPH_FILENAME="${GRAPH_FILENAME:-retweet_graph_5050_all_future_political_leaning.pt}"
GRAPH_ROOT="${GRAPH_ROOT:-/scratch1/eibl/data/midterm/graphs}"
FEATURE_SUBSET="${FEATURE_SUBSET:-stats_only}"
PREFIX_BASE="${PREFIX_BASE:-midterm_pl_icl}"
DATASET_LEN_CAP="${DATASET_LEN_CAP:-2000}"
VAL_LEN_CAP="${VAL_LEN_CAP:-500}"
TEST_LEN_CAP="${TEST_LEN_CAP:-500}"
EPOCHS="${EPOCHS:-3}"
EVAL_STEP="${EVAL_STEP:-100}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-250}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-0}"

case "$FEATURE_SUBSET" in
  stats_only)
    INPUT_DIM="${INPUT_DIM:-11}"
    ;;
  emb_only)
    INPUT_DIM="${INPUT_DIM:-384}"
    ;;
  all)
    INPUT_DIM="${INPUT_DIM:-395}"
    ;;
  constant1)
    INPUT_DIM="${INPUT_DIM:-1}"
    ;;
  keep:*)
    if [ -z "${INPUT_DIM:-}" ]; then
      echo "INPUT_DIM must be set when FEATURE_SUBSET=$FEATURE_SUBSET"
      exit 1
    fi
    ;;
  *)
    echo "Unsupported FEATURE_SUBSET=$FEATURE_SUBSET"
    echo "Use stats_only, emb_only, all, constant1, or keep:<...> with INPUT_DIM set."
    exit 1
    ;;
esac

if [ "$SHOT" -eq 0 ]; then
  ZERO_SHOT=True
else
  ZERO_SHOT=False
fi

PREFIX="${PREFIX_BASE}_${FEATURE_SUBSET}_shot${SHOT}"

echo "Running shot sweep task"
echo "  shot:           $SHOT"
echo "  zero_shot:      $ZERO_SHOT"
echo "  feature_subset: $FEATURE_SUBSET"
echo "  input_dim:      $INPUT_DIM"
echo "  graph:          ${GRAPH_ROOT}/${GRAPH_FILENAME}"
echo "  prefix:         $PREFIX"

python experiments/run_single_experiment.py \
  --dataset midterm \
  --root "$GRAPH_ROOT" \
  --graph_filename "$GRAPH_FILENAME" \
  --task classification \
  --midterm_feature_subset "$FEATURE_SUBSET" \
  --input_dim "$INPUT_DIM" \
  --original_features True \
  --n_way 2 \
  --n_shots "$SHOT" \
  --n_query 5 \
  --zero_shot "$ZERO_SHOT" \
  --dataset_len_cap "$DATASET_LEN_CAP" \
  --val_len_cap "$VAL_LEN_CAP" \
  --test_len_cap "$TEST_LEN_CAP" \
  --epochs "$EPOCHS" \
  --eval_step "$EVAL_STEP" \
  --checkpoint_step "$CHECKPOINT_STEP" \
  --workers "$WORKERS" \
  --device 0 \
  --seed "$SEED" \
  --prefix "$PREFIX"
