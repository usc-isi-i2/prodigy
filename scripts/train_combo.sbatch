
#!/bin/bash
# Two-phase combo training job: fine-tune an existing checkpoint on a second
# task, then evaluate the resulting checkpoint on the held-out third task.
#
# Submit via submit_train_combo.sh — do not submit directly.
#
# Required env vars:
#   DATASET       — midterm | ukr_rus_twitter | covid19_twitter
#   SRC_TASK      — task the source checkpoint was trained on (nm | lp | pl)
#   SRC_CKPT      — path to the source train1 checkpoint
#   FINETUNE_TASK — task to fine-tune on               (nm | lp | pl)
#   EVAL_TASK     — held-out task to evaluate on        (nm | lp | pl)
#
# Optional env vars:
#   SHOTS_CSV     — comma-separated shot counts for eval (default: 0,1,2,5,10)

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G        # overridden by submit wrapper via --mem=
#SBATCH --time=08:00:00  # covers fine-tune (~2h) + eval (~4h)

set -euo pipefail

: "${DATASET:?DATASET env var is required}"
: "${SRC_TASK:?SRC_TASK env var is required}"
: "${SRC_CKPT:?SRC_CKPT env var is required}"
: "${FINETUNE_TASK:?FINETUNE_TASK env var is required}"
: "${EVAL_TASK:?EVAL_TASK env var is required}"

SHOTS_CSV="${SHOTS_CSV:-0,1,2,5,10}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${SCRIPT_DIR}/config/${DATASET}.sh"
if [[ ! -f "$CONFIG" ]]; then
  echo "Unknown dataset '${DATASET}' — no config at ${CONFIG}" >&2
  exit 1
fi
source "$CONFIG"

module purge 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

cd /home1/eibl/gfm/prodigy
mkdir -p "$LOG_DIR"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -f "$SRC_CKPT" ]]; then
  echo "Source checkpoint not found: $SRC_CKPT" >&2
  exit 1
fi

# Normalise task tags
normalise_tag() {
  case "$1" in
    neighbor_matching|nm)        echo nm ;;
    temporal_link_prediction|lp) echo lp ;;
    classification|pl)           echo pl ;;
    *) echo "Unknown task: $1" >&2; exit 1 ;;
  esac
}
normalise_task() {
  case "$1" in
    nm) echo neighbor_matching ;;
    lp) echo temporal_link_prediction ;;
    pl) echo classification ;;
  esac
}

SRC_TAG="$(normalise_tag "$SRC_TASK")"
FT_TAG="$(normalise_tag "$FINETUNE_TASK")"
EVAL_TAG="$(normalise_tag "$EVAL_TASK")"

if [[ "$SRC_TAG" == "$FT_TAG" || "$SRC_TAG" == "$EVAL_TAG" || "$FT_TAG" == "$EVAL_TAG" ]]; then
  echo "SRC_TASK, FINETUNE_TASK, and EVAL_TASK must all be different." >&2
  exit 1
fi

# Check eval task is supported by this dataset
if [[ ! " ${SUPPORTED_TASKS[*]} " =~ " $(normalise_task "$EVAL_TAG") " ]]; then
  echo "Dataset '${DATASET}' does not support eval task '${EVAL_TAG}'." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Helper: build task args from config
# ---------------------------------------------------------------------------
task_args() {
  local tag="$1"
  local n_shots="$2"

  local zero_shot="$([[ "$n_shots" == "0" ]] && echo True || echo False)"

  case "$tag" in
    nm)
      echo "--n_way ${NM_N_WAY} --n_shots ${n_shots} --n_query ${NM_N_QUERY}"
      echo "--zero_shot ${zero_shot}"
      echo "--val_len_cap 500 --test_len_cap 500"
      echo "--eval_step 1000 --checkpoint_step 1000"
      ;;
    lp)
      echo "--n_way ${LP_N_WAY} --n_shots ${n_shots} --n_query ${LP_N_QUERY}"
      echo "--zero_shot ${zero_shot}"
      echo "--val_len_cap 500 --test_len_cap 500"
      echo "--epochs 3 --eval_step 1000 --checkpoint_step 1000"
      echo "--midterm_target_edge_view ${LP_TARGET_EDGE_VIEW}"
      ;;
    pl)
      echo "--n_way ${PL_N_WAY} --n_shots ${n_shots} --n_query ${PL_N_QUERY}"
      echo "--zero_shot ${zero_shot}"
      echo "--dataset_len_cap ${PL_DATASET_LEN_CAP}"
      echo "--val_len_cap 500 --test_len_cap 500"
      echo "--epochs 3 --eval_step 1000 --checkpoint_step 1000"
      echo "--midterm_label_downsample ${PL_LABEL_DOWNSAMPLE}"
      echo "--ignore_label_embeddings False --linear_probe False"
      ;;
  esac
}

task_graph() {
  case "$1" in
    nm) echo "${NM_GRAPH}" ;;
    lp) echo "${LP_GRAPH}" ;;
    pl) echo "${PL_GRAPH}" ;;
  esac
}

# Common args shared by all python calls
COMMON_ARGS=(
  --dataset               "$DATASET"
  --root                  "$GRAPH_ROOT"
  --midterm_feature_subset "$FEATURE_SUBSET"
  --input_dim             "$INPUT_DIM"
  --original_features     True
  --workers               "$WORKERS"
  --device 0
  --seed   0
)
if [[ -n "${EDGE_VIEW:-}" ]]; then
  COMMON_ARGS+=(--midterm_edge_view "$EDGE_VIEW")
fi

# ---------------------------------------------------------------------------
# Phase 1: Fine-tune the source checkpoint on FINETUNE_TASK
# ---------------------------------------------------------------------------
FT_PREFIX="combo_${DATASET}_${SRC_TAG}_ft_${FT_TAG}"
FT_TRAIN_SHOTS="$(eval echo "\${${FT_TAG^^}_TRAIN_SHOTS}")"

read -ra FT_EXTRA_ARGS <<< "$(task_args "$FT_TAG" "$FT_TRAIN_SHOTS")"

echo "========================================================"
echo "Phase 1: Fine-tune ${SRC_TAG} checkpoint on ${FT_TAG}"
echo "  src_ckpt : $SRC_CKPT"
echo "  prefix   : $FT_PREFIX"
echo "========================================================"

python3 experiments/run_single_experiment.py \
  "${COMMON_ARGS[@]}" \
  --graph_filename "$(task_graph "$FT_TAG")" \
  --task_name      "$(normalise_task "$FT_TAG")" \
  --pretrained_model_run "$SRC_CKPT" \
  --prefix         "$FT_PREFIX" \
  "${FT_EXTRA_ARGS[@]}"

# ---------------------------------------------------------------------------
# Locate the checkpoint produced by Phase 1
# ---------------------------------------------------------------------------
LATEST_STATE_DIR="$(ls -td state/${FT_PREFIX}_*/ 2>/dev/null | head -1)"
if [[ -z "$LATEST_STATE_DIR" ]]; then
  echo "Could not find a state directory matching prefix '${FT_PREFIX}'." >&2
  exit 1
fi

COMBO_CKPT="$(ls -t "${LATEST_STATE_DIR}checkpoint/"*.ckpt 2>/dev/null | head -1)"
if [[ -z "$COMBO_CKPT" ]]; then
  echo "No checkpoint (.ckpt) found under ${LATEST_STATE_DIR}checkpoint/" >&2
  exit 1
fi

echo ""
echo "Fine-tune checkpoint: $COMBO_CKPT"
echo ""

# ---------------------------------------------------------------------------
# Phase 2: Evaluate the combo checkpoint on EVAL_TASK across all shot counts
# ---------------------------------------------------------------------------
IFS=',' read -r -a SHOT_LIST <<< "$SHOTS_CSV"
EVAL_GRAPH="$(task_graph "$EVAL_TAG")"
EVAL_TASK_NAME="$(normalise_task "$EVAL_TAG")"
MODEL_NAME="combo_${DATASET}_${SRC_TAG}_ft_${FT_TAG}"

echo "========================================================"
echo "Phase 2: Evaluate ${MODEL_NAME} on ${EVAL_TAG}"
echo "  ckpt     : $COMBO_CKPT"
echo "  shots    : $SHOTS_CSV"
echo "========================================================"

for shots in "${SHOT_LIST[@]}"; do
  read -ra EVAL_EXTRA_ARGS <<< "$(task_args "$EVAL_TAG" "$shots")"

  echo "--- shots=${shots} ---"
  python3 experiments/run_single_experiment.py \
    "${COMMON_ARGS[@]}" \
    --graph_filename        "$EVAL_GRAPH" \
    --task_name             "$EVAL_TASK_NAME" \
    --pretrained_model_run  "$COMBO_CKPT" \
    --eval_only             True \
    --eval_test_before_train True \
    --eval_val_before_train  True \
    --save_roc_curve         True \
    --prefix "trained_on_${MODEL_NAME}_eval_on_${EVAL_TAG}_${shots}_shot" \
    "${EVAL_EXTRA_ARGS[@]}"
done

echo ""
echo "Combo experiment complete: ${SRC_TAG} → ${FT_TAG} → eval ${EVAL_TAG}"
