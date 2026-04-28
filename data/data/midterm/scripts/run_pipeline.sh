#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../../../" && pwd)"
cd "$ROOT_DIR"

MODEL="all-MiniLM-L6-v2"
EMB_PATH="data/data/midterm/embeddings/embeddings_${MODEL}.pt"
GRAPH_PATH="data/data/midterm/graphs/retweet_graph.pt"
CSV_GLOB="/project2/ll_774_951/midterm/*/*.csv"

python data/data/midterm/scripts/build_user_embeddings.py \
  --csv_glob "$CSV_GLOB" \
  --model "sentence-transformers/${MODEL}" \
  --out "$EMB_PATH"

python data/data/midterm/scripts/generate_user_graph.py \
  --csv_glob "$CSV_GLOB" \
  --embeddings "$EMB_PATH" \
  --embedding_pool meanpool \
  --out "$GRAPH_PATH"

python data/data/midterm/scripts/validate_graph.py --graph "$GRAPH_PATH"
python data/data/midterm/scripts/inspect_graph.py --graph "$GRAPH_PATH"
