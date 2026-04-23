# Dataset configuration: midterm
# Sourced by train_single_task.sbatch and eval_model_list.sbatch.

DATASET=midterm
GRAPH_ROOT=/scratch1/eibl/data/midterm/graphs
LOG_DIR=/home1/eibl/gfm/prodigy/logs
SLURM_MEM=32G
WORKERS=8
FEATURE_SUBSET=emb_only
INPUT_DIM=384
# EDGE_VIEW is intentionally unset — midterm uses the framework default.

SUPPORTED_TASKS=(neighbor_matching temporal_link_prediction classification)

# ---------------------------------------------------------------------------
# Per-task graph filenames
# ---------------------------------------------------------------------------
NM_GRAPH=retweet_graph_5050_all_future_political_leaning.pt
LP_GRAPH=retweet_graph_5050_all_future_political_leaning.pt
PL_GRAPH=retweet_graph_5050_all_future_political_leaning.pt

# ---------------------------------------------------------------------------
# Per-task defaults (used at train time; eval overrides shots via CLI)
# ---------------------------------------------------------------------------
NM_N_WAY=3;  NM_N_QUERY=24; NM_TRAIN_SHOTS=3
LP_N_WAY=1;  LP_N_QUERY=3;  LP_TRAIN_SHOTS=1
PL_N_WAY=2;  PL_N_QUERY=3;  PL_TRAIN_SHOTS=4

LP_TARGET_EDGE_VIEW=default
PL_LABEL_DOWNSAMPLE=50:50
PL_DATASET_LEN_CAP=2000
