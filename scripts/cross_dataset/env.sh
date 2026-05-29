#!/bin/bash
# Shared environment variables for cross-dataset experiments.
# Source this file at the start of each sbatch script:
#   source scripts/cross_dataset/env.sh

# Project paths
PROJECT_ROOT="/home1/singhama/gfm/prodigy"
LOGS_DIR="${PROJECT_ROOT}/logs"
STATE_DIR="${PROJECT_ROOT}/state"

# Conda environment
CONDA_ENV="/home1/singhama/.conda/envs/prodigy-env"

# Dataset roots
MIDTERM_ROOT="/scratch1/eibl/data/midterm/graphs"
COVID_ROOT="/scratch1/eibl/data/covid19_twitter/graphs"
UKR_RUS_ROOT="/scratch1/eibl/data/ukr_rus_twitter/graphs"
ELECTION2020_ROOT="/scratch1/eibl/data/election2020/graphs"
COVID_POLITICAL_ROOT="/scratch1/eibl/data/covid_political/graphs"
UKR_RUS_SUSPENDED_ROOT="/scratch1/eibl/data/ukr_rus_suspended/graphs"

# Graph filenames
MIDTERM_GRAPH="retweet_graph_1p5m.pt"
COVID_GRAPH="retweet_graph_1p5m_hf03_labeled.pt"
UKR_RUS_GRAPH="retweet_graph_1p5m_hf03_political_labels.pt"
ELECTION2020_GRAPH="retweet_graph.pt"
COVID_POLITICAL_GRAPH="retweet_graph.pt"
UKR_RUS_SUSPENDED_GRAPH="retweet_graph.pt"
