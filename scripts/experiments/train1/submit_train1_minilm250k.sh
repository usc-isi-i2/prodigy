#!/bin/bash
set -euo pipefail

REPO_DIR="/home1/eibl/gfm/prodigy"
LOG_DIR="${REPO_DIR}/logs/train1_minilm250k"
PARTITION="${PARTITION:-gpu}"
GRES="${GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-64G}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
DRY_RUN="${DRY_RUN:-0}"

MIDTERM_ROOT="/scratch1/eibl/data/midterm/graphs"
COVID19_ROOT="/scratch1/eibl/data/covid19_twitter/graphs"

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${LOG_DIR}"
fi

submit_job() {
  local job_name="$1"
  shift

  local -a cmd=(python3 experiments/run_single_experiment.py "$@")
  local cmd_str
  printf -v cmd_str '%q ' "${cmd[@]}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "=== ${job_name} ==="
    echo "${cmd_str}"
    echo
    return
  fi

  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${LOG_DIR}/${job_name}_%j.out
#SBATCH --error=${LOG_DIR}/${job_name}_%j.err
#SBATCH --partition=${PARTITION}
#SBATCH --gres=${GRES}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME_LIMIT}

set -euo pipefail

module purge || true
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

cd "${REPO_DIR}"
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}"

${cmd_str}
EOF
}

submit_midterm() {
  local graph_type="$1"   # retweet or interaction
  local graph_file="$2"
  local prefix="midterm_${graph_type}_minilm250k"

  submit_job "${prefix}_nm" \
    --dataset midterm \
    --root "${MIDTERM_ROOT}" \
    --graph_filename "${graph_file}" \
    --task_name neighbor_matching \
    --feature_subset emb_only \
    --edge_view temporal_history \
    --use_edge_features False \
    --original_features True \
    --n_way 3 \
    --n_shots 3 \
    --n_query 24 \
    --checkpoint_step 1000 \
    --val_len_cap 500 \
    --test_len_cap 500 \
    --eval_step 1000 \
    --workers 8 \
    --device 0 \
    --seed 0 \
    --prefix "train1_${prefix}_nm"

  submit_job "${prefix}_lp" \
    --dataset midterm \
    --root "${MIDTERM_ROOT}" \
    --graph_filename "${graph_file}" \
    --task_name temporal_link_prediction \
    --feature_subset emb_only \
    --edge_view temporal_history \
    --target_edge_view temporal_new \
    --use_edge_features False \
    --original_features True \
    --n_way 1 \
    --n_shots 1 \
    --n_query 3 \
    --zero_shot False \
    --val_len_cap 500 \
    --test_len_cap 500 \
    --epochs 3 \
    --eval_step 1000 \
    --checkpoint_step 1000 \
    --workers 8 \
    --device 0 \
    --seed 0 \
    --prefix "train1_${prefix}_lp"

  submit_job "${prefix}_pl" \
    --dataset midterm \
    --root "${MIDTERM_ROOT}" \
    --graph_filename "${graph_file}" \
    --task_name classification \
    --feature_subset emb_only \
    --use_edge_features False \
    --original_features True \
    --ignore_label_embeddings False \
    --linear_probe False \
    --n_way 2 \
    --n_shots 4 \
    --n_query 3 \
    --zero_shot False \
    --dataset_len_cap 2000 \
    --val_len_cap 500 \
    --test_len_cap 500 \
    --eval_step 1000 \
    --epochs 3 \
    --checkpoint_step 1000 \
    --workers 8 \
    --device 0 \
    --seed 0 \
    --midterm_label_downsample 50:50 \
    --prefix "train1_${prefix}_pl"
}

submit_covid19() {
  local graph_type="$1"   # retweet or interaction
  local graph_file="$2"
  local prefix="covid19_${graph_type}_minilm250k"

  submit_job "${prefix}_nm" \
    --dataset covid19_twitter \
    --root "${COVID19_ROOT}" \
    --graph_filename "${graph_file}" \
    --task_name neighbor_matching \
    --feature_subset emb_only \
    --edge_view temporal_history \
    --use_edge_features False \
    --original_features True \
    --n_way 3 \
    --n_shots 3 \
    --n_query 8 \
    --checkpoint_step 1000 \
    --val_len_cap 500 \
    --test_len_cap 500 \
    --eval_step 1000 \
    --workers 4 \
    --device 0 \
    --seed 0 \
    --prefix "train1_${prefix}_nm"

  submit_job "${prefix}_lp" \
    --dataset covid19_twitter \
    --root "${COVID19_ROOT}" \
    --graph_filename "${graph_file}" \
    --task_name temporal_link_prediction \
    --feature_subset emb_only \
    --edge_view temporal_history \
    --target_edge_view temporal_new \
    --use_edge_features False \
    --original_features True \
    --n_way 1 \
    --n_shots 1 \
    --n_query 3 \
    --zero_shot False \
    --val_len_cap 500 \
    --test_len_cap 500 \
    --epochs 3 \
    --eval_step 1000 \
    --checkpoint_step 1000 \
    --workers 4 \
    --device 0 \
    --seed 0 \
    --prefix "train1_${prefix}_lp"

  submit_job "${prefix}_pl" \
    --dataset covid19_twitter \
    --root "${COVID19_ROOT}" \
    --graph_filename "${graph_file}" \
    --task_name classification \
    --feature_subset emb_only \
    --use_edge_features False \
    --original_features True \
    --ignore_label_embeddings False \
    --linear_probe False \
    --n_way 2 \
    --n_shots 3 \
    --n_query 3 \
    --zero_shot False \
    --dataset_len_cap 2000 \
    --val_len_cap 500 \
    --test_len_cap 500 \
    --epochs 12 \
    --eval_step 1000 \
    --checkpoint_step 1000 \
    --workers 4 \
    --device 0 \
    --seed 0 \
    --midterm_label_downsample 50:50 \
    --prefix "train1_${prefix}_pl"
}

submit_midterm retweet     retweet_graph_minilm_250k.pt
submit_midterm interaction interaction_graph_minilm_250k.pt

submit_covid19 retweet     retweet_graph_minilm_250k_hf03_labeled.pt
submit_covid19 interaction interaction_graph_minilm_250k_hf03_labeled.pt
