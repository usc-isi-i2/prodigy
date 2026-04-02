# Ukraine-Russia Twitter Dataset

Retweet graph built from a Twitter dataset collected around the Ukraine-Russia conflict. Nodes are users, edges are retweet relationships. Structure and feature schema are identical to the midterm dataset.

## Raw data

CSV files organised as `<dataset_root>/*/*.csv`. On the cluster:

```
/project2/ll_774_951/uk_ru/twitter/data/*/*.csv
```

Each CSV row is one tweet and must contain at minimum: `userid`, `screen_name`, `rt_userid`, `rt_screen`, `date`, `followers_count`, `verified`, `statuses_count`.

## Pipeline

### Step 1 — Build user text embeddings

Encodes each user's tweet/bio text with a sentence transformer and mean-pools them into a single vector per user. Supports checkpointing (`--resume`) for large datasets.

```bash
python data/data/ukr_rus_twitter/scripts/build_user_embeddings.py \
  --csv_glob "/project2/ll_774_951/uk_ru/twitter/data/*/*.csv" \
  --model "sentence-transformers/all-MiniLM-L6-v2" \
  --out data/data/ukr_rus_twitter/embeddings/user_embeddings_minilm.pt \
  --checkpoint_path data/data/ukr_rus_twitter/embeddings/user_embeddings_minilm.checkpoint.pkl \
  --checkpoint_every 10 \
  --resume
```

| Argument | Default | Description |
|---|---|---|
| `--csv_glob` | `/project2/ll_774_951/uk_ru/twitter/data/*/*.csv` | Glob for raw CSV files |
| `--model` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace sentence-transformer model |
| `--out` | `data/data/ukr_rus_twitter/embeddings/user_embeddings_minilm.pt` | Output path |
| `--batch_size` | 256 | Encoding batch size |
| `--max_files` | 0 (all) | Limit number of CSV files |
| `--device` | auto | `cuda` or `cpu` |
| `--checkpoint_path` | `...minilm.checkpoint.pkl` | Intermediate checkpoint for resuming |
| `--checkpoint_every` | 5 | Save checkpoint every N files |
| `--resume` | off | Resume from existing checkpoint |

**Output:** a `.pt` file containing keys `user_ids` and `meanpool` (tensor `[num_users, emb_dim]`).

### Step 2 — Build the retweet graph

```bash
python data/data/ukr_rus_twitter/scripts/generate_retweet_graph.py \
  --csv "/project2/ll_774_951/uk_ru/twitter/data/*/*.csv" \
  --embeddings data/data/ukr_rus_twitter/embeddings/user_embeddings_minilm.pt \
  --embedding_pool meanpool \
  --history_fraction 0.3 \
  --out data/data/ukr_rus_twitter/graphs/retweet_graph.pt
```

| Argument | Default | Description |
|---|---|---|
| `--csv` | `/project2/ll_774_951/uk_ru/twitter/data/*/*.csv` | Glob for raw CSV files |
| `--out` | `data/data/ukr_rus_twitter/graphs/retweet_graph.pt` | Output path |
| `--embeddings` | `` (none) | Path to embeddings `.pt` from Step 1 |
| `--embedding_pool` | `meanpool` | `meanpool` or `maxpool` |
| `--history_fraction` | 0.8 | Fraction of edges in the "history" split for temporal LP |
| `--max_files` | 0 (all) | Limit number of CSV files |

**Output:** same schema as the midterm graph:

| Key | Shape | Description |
|---|---|---|
| `x` | `[N, F]` | Node features (11 numeric + optional 384-dim embeddings) |
| `edge_index` | `[2, E]` | Retweet edges |
| `edge_attr` | `[E, 4]` | `first_retweet_time`, `n_retweets`, `avg_rt_fav`, `avg_rt_reply` |
| `y` | `[N]` | Node labels |
| `user_ids` | `[N]` | User ID strings |
| `feature_names` | list | Names of the 11 numeric node features |
| `label_names` | list | Label class names |
| `temporal_*` | — | Temporal train/val/test edge masks |

Node features (11): `subscriber_count`, `verified`, `avg_favorites`, `avg_comments`, `avg_score`, `avg_n_hashtags`, `avg_n_mentions`, `avg_has_media`, `post_count`, `in_degree`, `out_degree`.

### Step 3 — Validate

```bash
python data/data/ukr_rus_twitter/scripts/validate_graph.py \
  --graph data/data/ukr_rus_twitter/graphs/retweet_graph.pt
```

## Running on the cluster

```bash
# Step 1: embeddings (GPU, ~90 min for 150 files)
sbatch data/data/ukr_rus_twitter/scripts/run_build_user_embeddings.sbatch

# Step 2: graph (CPU, ~60 min)
sbatch data/data/ukr_rus_twitter/scripts/run_generate_retweet_graph.sbatch
```

The `.sbatch` files have hardcoded output paths under `/scratch1/eibl/` — update those to your own scratch directory before submitting.

Experiment training scripts are in [`scripts/`](../../../scripts/):

```bash
sbatch scripts/submit_train1_ukr_rus_twitter_lp.sh   # temporal link prediction
sbatch scripts/submit_train1_ukr_rus_twitter_nm.sh   # neighbor matching
sbatch scripts/submit_train1_ukr_rus_twitter_pl.sh   # political leaning classification
```

## Experiments

| Task | `--task_name` | Key flags |
|---|---|---|
| Temporal link prediction | `temporal_link_prediction` | `--midterm_edge_view temporal_history --midterm_target_edge_view temporal_new --n_way 1 --n_shots 1 --n_query 3` |
| Neighbor matching | `neighbor_matching` | `--midterm_edge_view temporal_history --n_way 3 --n_shots 3 --n_query 24` |

Set `--midterm_feature_subset emb_only --input_dim 384` when using text embeddings only.
The graph filename used in the current scripts is `retweet_graph_150files_minilm_hf03.pt` (150 CSV files, MiniLM embeddings, history fraction 0.3).
