python -u data/data/covid19_twitter/scripts/generate_user_graph.py \
    --graph_mode retweet \
    --json_glob "/scratch1/eibl/data/covid19_twitter/raw/*/*.json" \
    --embeddings /scratch1/eibl/data/covid19_twitter/embeddings/user_embeddings_minilm_1p5m.pt \
    --embedding_pool meanpool \
    --max_nodes 250000 \
    --history_fraction 0.3 \
    --out /scratch1/eibl/data/covid19_twitter/graphs/retweet_graph_minilm_250k_hf03_labeled.pt \
&& python -u data/data/covid19_twitter/scripts/generate_user_graph.py \
    --graph_mode interaction \
    --json_glob "/scratch1/eibl/data/covid19_twitter/raw/*/*.json" \
    --embeddings /scratch1/eibl/data/covid19_twitter/embeddings/user_embeddings_minilm_1p5m.pt \
    --embedding_pool meanpool \
    --max_nodes 250000 \
    --history_fraction 0.3 \
    --out /scratch1/eibl/data/covid19_twitter/graphs/interaction_graph_minilm_250k_hf03_labeled.pt \
&& python -u data/data/midterm/scripts/generate_user_graph.py \
    --graph_mode retweet \
    --csv_glob "/project2/ll_774_951/midterm/*/*.csv" \
    --embeddings /scratch1/eibl/data/midterm/embeddings/user_embeddings_minilm_1p5m.pt \
    --embedding_pool meanpool \
    --max_nodes 250000 \
    --history_fraction 0.3 \
    --pseudo-label-margin 2 \
    --out /scratch1/eibl/data/midterm/graphs/retweet_graph_minilm_250k.pt \
&& python -u data/data/midterm/scripts/generate_user_graph.py \
    --graph_mode interaction \
    --csv_glob "/project2/ll_774_951/midterm/*/*.csv" \
    --embeddings /scratch1/eibl/data/midterm/embeddings/user_embeddings_minilm_1p5m.pt \
    --embedding_pool meanpool \
    --max_nodes 250000 \
    --history_fraction 0.3 \
    --pseudo-label-margin 2 \
    --out /scratch1/eibl/data/midterm/graphs/interaction_graph_minilm_250k.pt