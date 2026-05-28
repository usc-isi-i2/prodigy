#!/bin/bash
#SBATCH --job-name=generate_1p5m_graphs
#SBATCH --output=/home1/eibl/gfm/prodigy/logs/%x_%j.out
#SBATCH --error=/home1/eibl/gfm/prodigy/logs/%x_%j.err
#SBATCH --partition=main
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

module purge || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

cd /home1/eibl/gfm/prodigy
mkdir -p /home1/eibl/gfm/prodigy/logs

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Create output directories
mkdir -p /scratch1/eibl/data/covid_political/graphs
mkdir -p /scratch1/eibl/data/election2020/graphs
mkdir -p /scratch1/eibl/data/ukr_rus_suspended/graphs
mkdir -p /scratch1/eibl/data/covid19_twitter/graphs
mkdir -p /scratch1/eibl/data/ukr_rus_twitter/graphs
mkdir -p /scratch1/eibl/data/midterm/graphs

echo "=========================================="
echo "Generating 1p5m graphs for all datasets"
echo "=========================================="
echo ""

# 1. COVID Political (pickle + CSV, no max_nodes limit)
echo "[1/6] Generating covid_political retweet_graph.pt..."
python -u data/data/social_llm/scripts/generate_graph.py \
  --graph /project2/emiliofe_74/julie/social_llm_data/covid/graph.pickle \
  --csv /project2/emiliofe_74/julie/social_llm_data/covid/user_data.csv \
  --embeddings /scratch1/eibl/data/covid_political/embeddings/user_embeddings_minilm.pt \
  --out /scratch1/eibl/data/covid_political/graphs/retweet_graph.pt
echo "✓ COVID Political complete"
echo ""

# 2. Election 2020 (pickle + CSV, no max_nodes limit)
echo "[2/6] Generating election2020 retweet_graph.pt..."
python -u data/data/social_llm/scripts/generate_graph.py \
  --graph /project2/emiliofe_74/julie/social_llm_data/election2020/graph.pickle \
  --csv /project2/emiliofe_74/julie/social_llm_data/election2020/user_data.csv \
  --embeddings /scratch1/eibl/data/election2020/embeddings/user_embeddings_minilm.pt \
  --label_col label_conservative \
  --out /scratch1/eibl/data/election2020/graphs/retweet_graph.pt
echo "✓ Election 2020 complete"
echo ""

# 3. Ukraine-Russia Suspended (pickle + CSV, no max_nodes limit)
echo "[3/6] Generating ukr_rus_suspended retweet_graph.pt..."
python -u data/data/social_llm/scripts/generate_graph.py \
  --graph /project2/emiliofe_74/julie/social_llm_data/ukr_rus_suspended/graph.pickle \
  --csv /project2/emiliofe_74/julie/social_llm_data/ukr_rus_suspended/user_data.csv \
  --embeddings /scratch1/eibl/data/ukr_rus_suspended/embeddings/user_embeddings_minilm.pt \
  --label_col label_suspended \
  --out /scratch1/eibl/data/ukr_rus_suspended/graphs/retweet_graph.pt
echo "✓ Ukraine-Russia Suspended complete"
echo ""

# 4. COVID-19 Twitter (JSON, 1.5M nodes max)
echo "[4/6] Generating covid19_twitter retweet_graph_1p5m_hf03_labeled.pt..."
python -u data/data/covid19_twitter/scripts/generate_user_graph.py \
  --json_glob "/scratch1/eibl/data/covid19_twitter/raw/*/*.json" \
  --max_nodes 1500000 \
  --max_files 0 \
  --history_fraction 0.3 \
  --embeddings /scratch1/eibl/data/covid19_twitter/embeddings/user_embeddings_minilm.pt \
  --embedding_pool meanpool \
  --out /scratch1/eibl/data/covid19_twitter/graphs/retweet_graph_1p5m_hf03_labeled.pt
echo "✓ COVID-19 Twitter complete"
echo ""

# 5. Ukraine-Russia Twitter (CSV, 1.5M nodes max)
echo "[5/6] Generating ukr_rus_twitter retweet_graph_1p5m_hf03_political_labels.pt..."
python -u data/data/ukr_rus_twitter/scripts/generate_retweet_graph.py \
  --csv "/project2/ll_774_951/uk_ru/twitter/data/*/*.csv" \
  --max_nodes 1500000 \
  --max_files 0 \
  --history_fraction 0.3 \
  --embeddings /scratch1/eibl/data/ukr_rus_twitter/embeddings/user_embeddings_minilm.pt \
  --embedding_pool meanpool \
  --pseudo-political-labels \
  --out /scratch1/eibl/data/ukr_rus_twitter/graphs/retweet_graph_1p5m_hf03_political_labels.pt
echo "✓ Ukraine-Russia Twitter complete"
echo ""

# 6. Midterm (CSV, 1.5M nodes max)
echo "[6/6] Generating midterm retweet_graph_1p5m.pt..."
python -u data/data/midterm/scripts/generate_user_graph.py \
  --csv_glob "/project2/ll_774_951/midterm/*/*.csv" \
  --max_nodes 1500000 \
  --max_files 0 \
  --history_fraction 0.3 \
  --embeddings /scratch1/eibl/data/midterm/embeddings/embeddings_all-MiniLM-L6-v2.pt \
  --embedding_pool meanpool \
  --out /scratch1/eibl/data/midterm/graphs/retweet_graph_1p5m.pt
echo "✓ Midterm complete"
echo ""

echo "=========================================="
echo "All 1p5m graphs generated successfully!"
echo "=========================================="
