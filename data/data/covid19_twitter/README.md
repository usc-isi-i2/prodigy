# COVID-19 Twitter Dataset

Retweet graph built from a COVID-19 Twitter collection. Nodes are keyed by `userid`. Raw data is JSON or NDJSON. The graph builder can attach text embeddings, build temporal views, and attach external political labels from masking parquet files.

## Raw data

Cluster path:

```text
/scratch1/eibl/data/covid19_twitter/raw/*/*.json
```

The loader accepts:
- one JSON object per line
- JSON arrays
- wrapper dicts containing `statuses` or `data`

## Step 1: build user embeddings

```bash
python data/data/covid19_twitter/scripts/build_user_embeddings.py \
  --json_glob "/scratch1/eibl/data/covid19_twitter/raw/*/*.json" \
  --out /scratch1/eibl/data/covid19_twitter/embeddings/user_embeddings_minilm.pt
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--max_files` | limit number of JSON files |
| `--max_nodes` | cap the embedding artifact to exactly this many users when possible |
| `--stop_after_max_nodes / --no-stop_after_max_nodes` | stop at the cap or keep scanning for already-admitted users |
| `--batch_size` | encoder batch size |
| `--max_seq_len` | sentence-transformer max sequence length |

Embedding artifact keys:

| Key | Description |
|---|---|
| `user_ids` | canonical user ids |
| `handles` | aligned screen names, kept only as embedding metadata |
| `meanpool` | pooled embedding tensor `[N, D]` |
| `counts` | posts embedded per user |
| `model` | encoder name |

## Step 2: build the graph

```bash
python data/data/covid19_twitter/scripts/generate_user_graph.py \
  --json_glob "/scratch1/eibl/data/covid19_twitter/raw/*/*.json" \
  --embeddings /scratch1/eibl/data/covid19_twitter/embeddings/user_embeddings_minilm.pt \
  --embedding_pool meanpool \
  --history_fraction 0.3 \
  --out /scratch1/eibl/data/covid19_twitter/graphs/retweet_graph_hf03_labeled.pt
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--max_files` | limit number of JSON files |
| `--max_nodes` | trim the final graph to exactly this many nodes when possible |
| `--history_fraction` | temporal split fraction for `temporal_history` |
| `--future_target_mode` | `new_only` or `all_future` |
| `--labels_parquet_glob` | external label parquet glob, defaulting to `/scratch1/eibl/data/covid_masking/masking_2020-*.parquet` |
| `--keep-isolates / --no-keep-isolates` | keep or drop zero-degree nodes |

Graph artifact keys:

| Key | Description |
|---|---|
| `x` | node features |
| `edge_index` | directed user-user edges |
| `edge_attr` | edge features |
| `edge_attr_feature_names` | edge feature names |
| `user_ids` | canonical node ids |
| `u2i` | `user_id -> node_index` |
| `feature_names` | node feature names |
| `y` | external labels, `-1` for unlabeled |
| `label_names` | inferred from `political_gen` values |
| `edge_index_views` | includes `temporal_history` |
| `target_edge_index_views` | includes `temporal_new` |
| `future_edge_index` | LP target edges |
| `data` | compatibility `torch_geometric.data.Data` object |

## Step 3: validate and inspect

```bash
python data/data/covid19_twitter/scripts/validate_graph.py \
  --graph /scratch1/eibl/data/covid19_twitter/graphs/retweet_graph_hf03_labeled.pt
```

```bash
python data/data/covid19_twitter/scripts/inspect_graph.py \
  --graph /scratch1/eibl/data/covid19_twitter/graphs/retweet_graph_hf03_labeled.pt \
  --topk 20
```

## Training

```bash
sbatch scripts/submit_train1_covid19_twitter_pl.sh
sbatch scripts/submit_train1_covid19_twitter_nm.sh
sbatch scripts/submit_train1_covid19_twitter_lp.sh
```

Typical task settings used in recent comparable runs:

| Task | `--task_name` | Typical flags |
|---|---|---|
| Political labels | `classification` | `--n_way 2 --n_shots 3 --n_query 3 --midterm_label_downsample 50:50` |
| Neighbor matching | `neighbor_matching` | `--edge_view temporal_history --n_way 3 --n_shots 1 --n_query 12` |
| Temporal link prediction | `temporal_link_prediction` | `--edge_view temporal_history --target_edge_view temporal_new --n_way 1 --n_shots 1 --n_query 3` |

For embedding-only runs, use:

```text
--feature_subset emb_only --input_dim 384
```
