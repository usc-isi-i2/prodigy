# Dataset configuration: ukr_rus_twitter
# Sourced by train_single_task.sbatch and eval_model_list.sbatch.

DATASET=ukr_rus_twitter
GRAPH_ROOT=/scratch1/eibl/data/ukr_rus_twitter/graphs
LOG_DIR=/scratch1/eibl/data/ukr_rus_twitter/logs
SLURM_MEM=64G
WORKERS=4
FEATURE_SUBSET=emb_only
INPUT_DIM=384
EDGE_VIEW=temporal_history

SUPPORTED_TASKS=(neighbor_matching temporal_link_prediction classification)

# ---------------------------------------------------------------------------
# Per-task graph filenames
# (classification uses a separate graph with political labels attached)
# ---------------------------------------------------------------------------
NM_GRAPH=retweet_graph_150files_minilm_hf03.pt
LP_GRAPH=retweet_graph_150files_minilm_hf03.pt
PL_GRAPH=retweet_graph_150files_minilm_hf03_political_labels.pt

# ---------------------------------------------------------------------------
# Per-task defaults (used at train time; eval overrides shots via CLI)
# ---------------------------------------------------------------------------
NM_N_WAY=2;  NM_N_QUERY=8;  NM_TRAIN_SHOTS=3
LP_N_WAY=1;  LP_N_QUERY=3;  LP_TRAIN_SHOTS=1
PL_N_WAY=2;  PL_N_QUERY=4;  PL_TRAIN_SHOTS=0  # zero-shot at train time

LP_TARGET_EDGE_VIEW=temporal_new
PL_LABEL_DOWNSAMPLE=50:50
PL_DATASET_LEN_CAP=2000
