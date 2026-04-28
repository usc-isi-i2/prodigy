#!/bin/bash
set -euo pipefail

REPO_DIR="/home1/eibl/gfm/prodigy"
LOG_DIR="${REPO_DIR}/logs/interaction_feature_ablation"
PARTITION="${PARTITION:-gpu}"
GRES="${GRES:-gpu:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-64G}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
DRY_RUN="${DRY_RUN:-0}"

FEATURE_SUBSETS=(constant1 emb_only stats_only all)

MIDTERM_ROOT="/scratch1/eibl/data/midterm/graphs"
MIDTERM_GRAPH="interaction_graph_minilm_250k.pt"

COVID19_ROOT="/scratch1/eibl/data/covid19_twitter/graphs"
COVID19_GRAPH="interaction_graph_minilm_250k_hf03_labeled.pt"

COVID_POL_ROOT="/scratch1/eibl/data/covid_political/graphs"
COVID_POL_GRAPH="retweet_graph.pt"

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

submit_midterm_runs() {
  local feature_subset="$1"

  submit_job "midterm_nm_${feature_subset}" \
    --dataset midterm \
    --root "${MIDTERM_ROOT}" \
    --graph_filename "${MIDTERM_GRAPH}" \
    --task_name neighbor_matching \
    --feature_subset "${feature_subset}" \
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
    --prefix "train1_midterm_nm_interaction_250k_${feature_subset}"

  submit_job "midterm_lp_${feature_subset}" \
    --dataset midterm \
    --root "${MIDTERM_ROOT}" \
    --graph_filename "${MIDTERM_GRAPH}" \
    --task_name temporal_link_prediction \
    --feature_subset "${feature_subset}" \
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
    --prefix "train1_midterm_lp_interaction_250k_${feature_subset}"

  submit_job "midterm_pl_${feature_subset}" \
    --dataset midterm \
    --root "${MIDTERM_ROOT}" \
    --graph_filename "${MIDTERM_GRAPH}" \
    --task_name classification \
    --feature_subset "${feature_subset}" \
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
    --prefix "train1_midterm_pl_interaction_250k_${feature_subset}"
}

submit_covid19_runs() {
  local feature_subset="$1"

  submit_job "covid19_nm_${feature_subset}" \
    --dataset covid19_twitter \
    --root "${COVID19_ROOT}" \
    --graph_filename "${COVID19_GRAPH}" \
    --task_name neighbor_matching \
    --feature_subset "${feature_subset}" \
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
    --prefix "train1_covid19_twitter_nm_interaction_250k_${feature_subset}"

  submit_job "covid19_lp_${feature_subset}" \
    --dataset covid19_twitter \
    --root "${COVID19_ROOT}" \
    --graph_filename "${COVID19_GRAPH}" \
    --task_name temporal_link_prediction \
    --feature_subset "${feature_subset}" \
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
    --prefix "train1_covid19_twitter_lp_interaction_250k_${feature_subset}"

  submit_job "covid19_pl_${feature_subset}" \
    --dataset covid19_twitter \
    --root "${COVID19_ROOT}" \
    --graph_filename "${COVID19_GRAPH}" \
    --task_name classification \
    --feature_subset "${feature_subset}" \
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
    --prefix "train1_covid19_twitter_pl_interaction_250k_${feature_subset}"
}

submit_covid_political_runs() {
  local feature_subset="$1"

  submit_job "covidpol_nm_${feature_subset}" \
    --dataset covid_political \
    --root "${COVID_POL_ROOT}" \
    --graph_filename "${COVID_POL_GRAPH}" \
    --task_name neighbor_matching \
    --feature_subset "${feature_subset}" \
    --use_edge_features False \
    --original_features True \
    --n_way 3 \
    --n_shots 3 \
    --n_query 8 \
    --checkpoint_step 1000 \
    --val_len_cap 500 \
    --test_len_cap 500 \
    --dataset_len_cap 2000 \
    --epochs 20 \
    --eval_step 1000 \
    --workers 4 \
    --device 0 \
    --seed 0 \
    --prefix "train1_covid_political_nm_${feature_subset}"

  submit_job "covidpol_pl_${feature_subset}" \
    --dataset covid_political \
    --root "${COVID_POL_ROOT}" \
    --graph_filename "${COVID_POL_GRAPH}" \
    --task_name classification \
    --feature_subset "${feature_subset}" \
    --use_edge_features False \
    --original_features True \
    --ignore_label_embeddings False \
    --linear_probe False \
    --n_way 2 \
    --n_shots 3 \
    --n_query 8 \
    --checkpoint_step 1000 \
    --val_len_cap 500 \
    --test_len_cap 500 \
    --dataset_len_cap 2000 \
    --epochs 20 \
    --eval_step 1000 \
    --workers 4 \
    --device 0 \
    --seed 0 \
    --midterm_label_downsample 50:50 \
    --prefix "train1_covid_political_pl_${feature_subset}"
}

for feature_subset in "${FEATURE_SUBSETS[@]}"; do
  submit_midterm_runs "${feature_subset}"
  submit_covid19_runs "${feature_subset}"
  submit_covid_political_runs "${feature_subset}"
done
