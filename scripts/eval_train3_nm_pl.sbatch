#!/bin/bash
#SBATCH --job-name=eval_train3_nm_pl
#SBATCH --output=/home1/eibl/gfm/prodigy/logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail

module purge || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

cd /home1/eibl/gfm/prodigy
mkdir -p /home1/eibl/gfm/prodigy/logs
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: sbatch $0 <model_list.txt> <dataset> [shots_csv] [tags_csv]" >&2
  echo "Datasets: covid19_twitter midterm ukr_rus_twitter covid_political election2020" >&2
  echo "          ukr_rus_suspended hate_bots05 hate_bots08 covid_mf ukr_rus_hate" >&2
  exit 1
fi

MODEL_LIST="$1"
DATASET="$2"
SHOTS_CSV="${3:-0,3}"

if [[ ! -f "$MODEL_LIST" ]]; then
  echo "Model list not found: $MODEL_LIST" >&2
  exit 1
fi

IFS=',' read -r -a SHOT_LIST <<< "$SHOTS_CSV"

TAGS_ARGS=()
if [[ -n "${4:-}" ]]; then
  IFS=',' read -r -a _tag_list <<< "$4"
  TAGS_ARGS=(--tags "${_tag_list[@]}")
fi

EDGE_VIEW_ARGS=()
case "$DATASET" in
  covid19_twitter)
    ROOT=/scratch1/eibl/data/covid19_twitter/graphs
    GRAPH_FILE=retweet_graph_1p5m_hf03_labeled.pt
    EDGE_VIEW_ARGS=(--edge_view temporal_history)
    VAL_LEN_CAP=1000; TEST_LEN_CAP=1000; DOWNSAMPLE=true
    ;;
  midterm)
    ROOT=/scratch1/eibl/data/midterm/graphs
    GRAPH_FILE=retweet_graph_1p5m.pt
    VAL_LEN_CAP=1000; TEST_LEN_CAP=1000; DOWNSAMPLE=true
    ;;
  ukr_rus_twitter)
    ROOT=/scratch1/eibl/data/ukr_rus_twitter/graphs
    GRAPH_FILE=retweet_graph_1p5m_hf03_political_labels.pt
    EDGE_VIEW_ARGS=(--edge_view temporal_history)
    VAL_LEN_CAP=1000; TEST_LEN_CAP=1000; DOWNSAMPLE=true
    ;;
  covid_political)
    ROOT=/scratch1/eibl/data/covid_political/graphs
    GRAPH_FILE=retweet_graph.pt
    VAL_LEN_CAP=500; TEST_LEN_CAP=500; DOWNSAMPLE=true
    ;;
  election2020)
    ROOT=/scratch1/eibl/data/election2020/graphs
    GRAPH_FILE=retweet_graph.pt
    VAL_LEN_CAP=500; TEST_LEN_CAP=500; DOWNSAMPLE=false
    ;;
  ukr_rus_suspended)
    ROOT=/scratch1/eibl/data/ukr_rus_suspended/graphs
    GRAPH_FILE=retweet_graph.pt
    VAL_LEN_CAP=500; TEST_LEN_CAP=500; DOWNSAMPLE=false
    ;;
  hate_bots05)
    ROOT=/home1/eibl/gfm/prodigy/data/data/hate_bots05/graphs
    GRAPH_FILE=retweet_graph.pt
    VAL_LEN_CAP=1000; TEST_LEN_CAP=1000; DOWNSAMPLE=false
    ;;
  hate_bots08)
    ROOT=/home1/eibl/gfm/prodigy/data/data/hate_bots08/graphs
    GRAPH_FILE=retweet_graph.pt
    VAL_LEN_CAP=1000; TEST_LEN_CAP=1000; DOWNSAMPLE=false
    ;;
  covid_mf)
    ROOT=/home1/eibl/gfm/prodigy/data/data/covid_mf/graphs
    GRAPH_FILE=retweet_graph.pt
    VAL_LEN_CAP=1000; TEST_LEN_CAP=1000; DOWNSAMPLE=false
    ;;
  ukr_rus_hate)
    ROOT=/home1/eibl/gfm/prodigy/data/data/ukr_rus_hate/graphs
    GRAPH_FILE=retweet_graph.pt
    VAL_LEN_CAP=1000; TEST_LEN_CAP=1000; DOWNSAMPLE=false
    ;;
  *)
    echo "Unknown dataset: $DATASET" >&2
    exit 1
    ;;
esac

COMMON_ARGS=(
  --dataset "$DATASET"
  --root "$ROOT"
  --graph_filename "$GRAPH_FILE"
  --feature_subset emb_only
  "${EDGE_VIEW_ARGS[@]}"
  --input_dim 384
  --original_features True
  --workers 4
  --device 0
  --seed 0
  --eval_only True
  --eval_test_before_train True
  --eval_val_before_train True
  --save_roc_curve True
)

run_eval() {
  local model_name="$1"
  local ckpt_path="$2"
  local task_name="$3"
  local shots="$4"
  local zero_shot
  zero_shot="$([[ "$shots" == "0" ]] && echo True || echo False)"

  local task_tag extra_args=()
  case "$task_name" in
    neighbor_matching)
      task_tag="nm"
      extra_args=(
        --task_name neighbor_matching
        --n_way 3
        --n_shots "$shots"
        --n_query 12
        --zero_shot "$zero_shot"
        --dataset_len_cap 5000
        --val_len_cap "$VAL_LEN_CAP"
        --test_len_cap "$TEST_LEN_CAP"
        --epochs 12
        --eval_step 2000
        --checkpoint_step 2000
        --prefix "eval_${model_name}_to_${DATASET}_${task_tag}_${shots}shot"
      )
      ;;
    classification)
      task_tag="pl"
      local pl_extra=()
      [[ "$DOWNSAMPLE" == "true" ]] && pl_extra=(--midterm_label_downsample 50:50)
      extra_args=(
        --task_name classification
        --ignore_label_embeddings False
        --linear_probe False
        --n_way 2
        --n_shots "$shots"
        --n_query 12
        --zero_shot "$zero_shot"
        --dataset_len_cap 2000
        --val_len_cap "$VAL_LEN_CAP"
        --test_len_cap "$TEST_LEN_CAP"
        --epochs 12
        --eval_step 2000
        --checkpoint_step 2000
        "${pl_extra[@]}"
        --prefix "eval_${model_name}_to_${DATASET}_${task_tag}_${shots}shot"
      )
      ;;
    *)
      echo "Unknown task: $task_name" >&2
      exit 1
      ;;
  esac

  echo "=== Evaluating model=${model_name} task=${task_name} shots=${shots} ckpt=${ckpt_path} ==="
  python3 experiments/run_single_experiment.py \
    "${COMMON_ARGS[@]}" \
    --pretrained_model_run "$ckpt_path" \
    "${extra_args[@]}" \
    "${TAGS_ARGS[@]}"
}

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="${raw_line#"${raw_line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"
  [[ -z "$line" || "$line" == \#* ]] && continue

  read -r first second extra <<< "$line"
  if [[ -n "${second:-}" ]]; then
    model_name="$first"
    ckpt_path="$second"
    [[ -n "${extra:-}" ]] && { echo "Invalid line in ${MODEL_LIST}: '$raw_line'" >&2; exit 1; }
  else
    ckpt_path="$first"
    model_name="$(basename "$(dirname "$ckpt_path")")"
  fi

  if [[ ! -f "$ckpt_path" ]]; then
    echo "Checkpoint not found for model '${model_name}': $ckpt_path" >&2
    exit 1
  fi

  for shots in "${SHOT_LIST[@]}"; do
    run_eval "$model_name" "$ckpt_path" neighbor_matching "$shots"
    run_eval "$model_name" "$ckpt_path" classification "$shots"
  done
done < "$MODEL_LIST"
