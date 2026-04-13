# Dataset configuration: covid19_twitter
# Sourced by train_single_task.sbatch and eval_model_list.sbatch.

DATASET=covid19_twitter
GRAPH_ROOT=/scratch1/eibl/data/covid19_twitter/graphs
LOG_DIR=/scratch1/eibl/data/covid19_twitter/logs
SLURM_MEM=64G
WORKERS=4
FEATURE_SUBSET=emb_only
INPUT_DIM=384
EDGE_VIEW=temporal_history

# covid19_twitter has no political-label graph, so classification is excluded.
SUPPORTED_TASKS=(neighbor_matching temporal_link_prediction)

# ---------------------------------------------------------------------------
# Per-task graph filenames
# ---------------------------------------------------------------------------
NM_GRAPH=retweet_graph_minilm_first100_hf03.pt
LP_GRAPH=retweet_graph_minilm_first100_hf03.pt

# ---------------------------------------------------------------------------
# Per-task defaults (used at train time; eval overrides shots via CLI)
# ---------------------------------------------------------------------------
NM_N_WAY=2;  NM_N_QUERY=8;  NM_TRAIN_SHOTS=3
LP_N_WAY=1;  LP_N_QUERY=3;  LP_TRAIN_SHOTS=1

LP_TARGET_EDGE_VIEW=temporal_new
