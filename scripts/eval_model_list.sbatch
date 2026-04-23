#!/bin/bash
# Generic model-list evaluation job.
# Submit via submit_eval.sh — do not submit directly.
#
# Required env vars:
#   DATASET      — one of: midterm, ukr_rus_twitter, covid19_twitter
#   MODEL_LIST   — path to a text file of checkpoints (see format below)
#
# Optional env vars:
#   SHOTS_CSV    — comma-separated shot counts (default: 0,1,2,5,10)
#
# Model list format — one checkpoint per line, either:
#   <ckpt_path>
#   <model_name> <ckpt_path>
# Lines starting with # are ignored.

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G        # overridden by submit wrapper via --mem=
#SBATCH --time=04:00:00

set -euo pipefail

: "${DATASET:?DATASET env var is required}"
: "${MODEL_LIST:?MODEL_LIST env var is required}"

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

if [[ ! -f "$MODEL_LIST" ]]; then
  echo "Model list not found: $MODEL_LIST" >&2
  exit 1
fi

IFS=',' read -r -a SHOT_LIST <<< "$SHOTS_CSV"
if [[ ${#SHOT_LIST[@]} -eq 0 ]]; then
  echo "Empty shot list from SHOTS_CSV='${SHOTS_CSV}'" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# run_eval <model_name> <ckpt_path> <task_tag> <shots>
# ---------------------------------------------------------------------------
run_eval() {
  local model_name="$1"
  local ckpt_path="$2"
  local tag="$3"
  local shots="$4"

  local task graph_filename n_way n_query task_args
  local zero_shot="$([[ "$shots" == "0" ]] && echo True || echo False)"

  case "$tag" in
    nm)
      task=neighbor_matching
      graph_filename="${NM_GRAPH}"
      n_way="${NM_N_WAY}"; n_query="${NM_N_QUERY}"
      task_args=(
        --n_way "$n_way" --n_shots "$shots" --n_query "$n_query"
        --zero_shot "$zero_shot"
        --val_len_cap 500 --test_len_cap 500
        --eval_step 1000 --checkpoint_step 1000
      )
      ;;
    lp)
      task=temporal_link_prediction
      graph_filename="${LP_GRAPH}"
      n_way="${LP_N_WAY}"; n_query="${LP_N_QUERY}"
      task_args=(
        --n_way "$n_way" --n_shots "$shots" --n_query "$n_query"
        --zero_shot "$zero_shot"
        --val_len_cap 500 --test_len_cap 500
        --epochs 3 --eval_step 1000 --checkpoint_step 1000
        --midterm_target_edge_view "${LP_TARGET_EDGE_VIEW}"
      )
      ;;
    pl)
      task=classification
      graph_filename="${PL_GRAPH}"
      n_way="${PL_N_WAY}"; n_query="${PL_N_QUERY}"
      task_args=(
        --n_way "$n_way" --n_shots "$shots" --n_query "$n_query"
        --zero_shot "$zero_shot"
        --dataset_len_cap "${PL_DATASET_LEN_CAP}"
        --val_len_cap 500 --test_len_cap 500
        --epochs 3 --eval_step 1000 --checkpoint_step 1000
        --midterm_label_downsample "${PL_LABEL_DOWNSAMPLE}"
        --ignore_label_embeddings False --linear_probe False
      )
      ;;
    *) echo "Unknown task tag: $tag" >&2; return 1 ;;
  esac

  local common_args=(
    --dataset        "$DATASET"
    --root           "$GRAPH_ROOT"
    --graph_filename "$graph_filename"
    --task_name      "$task"
    --midterm_feature_subset "$FEATURE_SUBSET"
    --input_dim      "$INPUT_DIM"
    --original_features True
    --workers        "$WORKERS"
    --device 0
    --seed   0
    --eval_only True
    --eval_test_before_train True
    --eval_val_before_train True
    --save_roc_curve True
    --pretrained_model_run "$ckpt_path"
    --prefix "trained_on_${model_name}_eval_on_${tag}_${shots}_shot"
  )

  if [[ -n "${EDGE_VIEW:-}" ]]; then
    common_args+=(--midterm_edge_view "$EDGE_VIEW")
  fi

  echo "=== Eval: model=${model_name} task=${task} shots=${shots} ==="
  python3 experiments/run_single_experiment.py "${common_args[@]}" "${task_args[@]}"
}

# ---------------------------------------------------------------------------
# Parse model list and dispatch
# ---------------------------------------------------------------------------
while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  # Trim leading/trailing whitespace
  line="${raw_line#"${raw_line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"
  [[ -z "$line" || "$line" == \#* ]] && continue

  read -r first second extra <<< "$line"
  if [[ -n "${second:-}" ]]; then
    [[ -n "${extra:-}" ]] && {
      echo "Invalid model list line (expected 1–2 fields): '${raw_line}'" >&2
      exit 1
    }
    model_name="$first"
    ckpt_path="$second"
  else
    ckpt_path="$first"
    model_name="$(basename "$(dirname "$ckpt_path")")"
  fi

  if [[ ! -f "$ckpt_path" ]]; then
    echo "Checkpoint not found for '${model_name}': ${ckpt_path}" >&2
    exit 1
  fi

  for shots in "${SHOT_LIST[@]}"; do
    for task_entry in "${SUPPORTED_TASKS[@]}"; do
      case "$task_entry" in
        neighbor_matching)        run_eval "$model_name" "$ckpt_path" nm "$shots" ;;
        temporal_link_prediction) run_eval "$model_name" "$ckpt_path" lp "$shots" ;;
        classification)           run_eval "$model_name" "$ckpt_path" pl "$shots" ;;
      esac
    done
  done
done < "$MODEL_LIST"
