# COVID-19 Twitter Dataset

Retweet graph built from a COVID-19 Twitter collection. Nodes are users, directed edges are retweet relationships. Raw data is JSON, typically newline-delimited JSON (one tweet object per line), though the loaders also accept JSON arrays and wrapper dicts.

## Raw data

JSON files are expected under:

```bash
/scratch1/eibl/data/covid19_twitter/raw/*/*.json
```

Example layout:

```text
/scratch1/eibl/data/covid19_twitter/raw/
  coronavirus-raw-2020-01-23/
    coronavirus-raw-2020-01-23-18.json
    coronavirus-raw-2020-01-23-19.json
  coronavirus-raw-2020-01-24/
    ...
```

Each tweet JSON object should contain at least:
- `created_at`
- `user.screen_name`
- `user.id`

Retweet edges are extracted from:
- `retweeted_status.user.screen_name`
- `retweeted_status.user.id`

## Pipeline

### Step 1 — Build user text embeddings

Build one pooled MiniLM embedding per handle from tweet text plus profile description. The script supports checkpointing and resume for large runs.

```bash
python data/data/covid19_twitter/scripts/build_user_embeddings.py \
  --json_glob "/scratch1/eibl/data/covid19_twitter/raw/*/*.json" \
  --model "sentence-transformers/all-MiniLM-L6-v2" \
  --out data/data/covid19_twitter/embeddings/user_embeddings_minilm.pt \
  --checkpoint_path data/data/covid19_twitter/embeddings/user_embeddings_minilm.checkpoint.pkl \
  --checkpoint_every 5 \
  --resume
```

Useful options:

| Argument | Default | Description |
|---|---|---|
| `--json_glob` | `/scratch1/eibl/data/covid19_twitter/raw/*/*.json` | Glob for raw JSON files |
| `--model` | `sentence-transformers/all-MiniLM-L6-v2` | SentenceTransformer model |
| `--out` | `data/data/covid19_twitter/embeddings/user_embeddings_minilm.pt` | Output embeddings file |
| `--batch_size` | `256` | Encoding batch size |
| `--max_files` | `0` | Limit number of files for sanity runs |
| `--device` | auto | `cuda` or `cpu` |
| `--checkpoint_path` | `...user_embeddings_minilm.checkpoint.pkl` | Checkpoint state for resume |
| `--checkpoint_every` | `5` | Save checkpoint every N processed files |
| `--resume` | off | Resume from existing checkpoint |

Output schema:

| Key | Description |
|---|---|
| `handles` | sorted list of lowercased screen names |
| `meanpool` | tensor `[N, D]` |
| `maxpool` | tensor `[N, D]` |
| `counts` | dict `handle -> number of embedded posts` |
| `model` | model name string |

### Step 2 — Build the retweet graph

Construct the graph and optionally attach the embeddings from Step 1.

```bash
python data/data/covid19_twitter/scripts/generate_retweet_graph.py \
  --json_glob "/scratch1/eibl/data/covid19_twitter/raw/*/*.json" \
  --embeddings data/data/covid19_twitter/embeddings/user_embeddings_minilm.pt \
  --embedding_pool meanpool \
  --history_fraction 0.3 \
  --out data/data/covid19_twitter/graphs/retweet_graph_minilm_hf03.pt
```

Useful options:

| Argument | Default | Description |
|---|---|---|
| `--json_glob` | `/scratch1/eibl/data/covid19_twitter/raw/*/*.json` | Glob for raw JSON files |
| `--out` | `data/data/covid19_twitter/graphs/retweet_graph.pt` | Output graph path |
| `--embeddings` | empty | Optional embeddings file from Step 1 |
| `--embedding_pool` | `meanpool` | `meanpool` or `maxpool` |
| `--history_fraction` | `0.8` | Fraction of rows used as history for temporal LP |
| `--future_target_mode` | `new_only` | Future LP targets: unseen only vs all future |
| `--max_files` | `0` | Limit number of JSON files |
| `--strict_dates` | off | Fail on bad timestamps instead of dropping them |
| `--keep-isolates` | false | Keep nodes with zero in-degree and out-degree |

Current graph artifact schema:

| Key | Shape | Description |
|---|---|---|
| `x` | `[N, F]` | Node features |
| `edge_index` | `[2, E]` | Directed retweet edges |
| `edge_attr` | `[E, 4]` | Edge features |
| `handles` | `list[str]` | Node IDs |
| `h2i` | `dict[str, int]` | Handle-to-index map |
| `feature_names` | list | Node feature names |
| `y` | `[N]` | Labels, currently all `-1` |
| `label_names` | list | Currently empty |
| `edge_index_views` | dict | Includes `temporal_history` |
| `target_edge_index_views` | dict | Includes `temporal_new` |
| `future_edge_index` | `[2, E_future]` | LP target edges |
| `data` | PyG `Data` | Convenience copy for compatibility |

Node features:
- 11 graph/account stats:
  `subscriber_count`, `verified`, `avg_favorites`, `avg_comments`, `avg_score`,
  `avg_n_hashtags`, `avg_n_mentions`, `avg_has_media`, `post_count`, `in_degree`, `out_degree`
- optional 384-dim MiniLM embeddings appended as `emb_0 ... emb_383`

Edge features:
- `first_retweet_time`
- `n_retweets`
- `avg_rt_fav`
- `avg_rt_reply`

### Step 3 — Validate and inspect

```bash
python data/data/covid19_twitter/scripts/validate_graph.py \
  --graph data/data/covid19_twitter/graphs/retweet_graph_minilm_hf03.pt
```

```bash
python data/data/covid19_twitter/scripts/inspect_graph.py \
  --graph data/data/covid19_twitter/graphs/retweet_graph_minilm_hf03.pt \
  --topk 20
```

## Running on the cluster

```bash
# Step 1: embeddings
sbatch data/data/covid19_twitter/scripts/run_build_user_embeddings.sbatch

# Step 2: graph
sbatch data/data/covid19_twitter/scripts/run_generate_retweet_graph.sbatch
```

These `.sbatch` files currently use hardcoded paths under `/scratch1/eibl/` and `/home1/eibl/`.

Notes:
- `run_build_user_embeddings.sbatch` is configured for the raw JSON path and writes to `/scratch1/eibl/data/covid19_twitter/embeddings/`
- `run_generate_retweet_graph.sbatch` is currently configured for the first `100` files with `history_fraction 0.3`, outputting `retweet_graph_minilm_first100_hf03.pt`

## Training

Available task-1 scripts:

```bash
sbatch scripts/submit_train1_covid19_twitter_nm.sh
sbatch scripts/submit_train1_covid19_twitter_lp.sh
```

Supported tasks right now:

| Task | `--task_name` | Notes |
|---|---|---|
| Neighbor matching | `neighbor_matching` | supported |
| Temporal link prediction | `temporal_link_prediction` | supported |
| Classification | `classification` | not supported until labels are added |

The current training scripts assume:
- `--midterm_feature_subset emb_only`
- `--input_dim 384`

If you are using the first-100-file sanity graph, make sure the script `--graph_filename` matches the graph you actually built, for example:

```text
retweet_graph_minilm_first100_hf03.pt
```
