#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/augment_midterm_graph_temporal_edge_features.py \
  --source-graph midterm/graph/graph_data.pt \
  --output-graph midterm/graph/graph_data.pt \
  --edge-view-name temporal_history \
  --target-view-name temporal_new
