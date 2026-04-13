# Midterm Twitter Dataset

Retweet graph built from the 2022 US midterm election Twitter dataset. Nodes are keyed by `userid`. The graph builder can attach text embeddings, build temporal views for link prediction, and generate weak political labels.

## Raw data

Cluster path:

```text
/project2/ll_774_951/midterm/*/*.csv
```

Each CSV row is one tweet. The graph builder expects retweet-related columns such as `userid`, `rt_userid`, `date`, `followers_count`, `verified`, `statuses_count`, `rt_fav_count`, `rt_reply_count`, `description`, `urls` or `urls_list`, `hashtag`, `rt_hashtag`, `mentionsn`, and `media_urls`.

## Step 1: build user embeddings

```bash
python data/data/midterm/scripts/build_user_embeddings.py \
  --csv_glob "/project2/ll_774_951/midterm/*/*.csv" \
  --out /scratch1/eibl/data/midterm/embeddings/user_embeddings_minilm.pt
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--max_files` | limit number of CSV files |
| `--max_nodes` | cap the embedding artifact to exactly this many users when possible |
| `--stop_after_max_nodes / --no-stop_after_max_nodes` | stop reading files once the cap is reached, or keep scanning later files for already-admitted users |
| `--batch_size` | encoder batch size |
| `--max_seq_len` | sentence-transformer max sequence length |

Embedding artifact keys:

| Key | Description |
|---|---|
| `user_ids` | canonical node ids |
| `meanpool` | pooled embedding tensor `[N, D]` |
| `counts` | posts embedded per user |
| `model` | encoder name |

## Step 2: build the graph

```bash
python data/data/midterm/scripts/generate_retweet_graph.py \
  --csv_glob "/project2/ll_774_951/midterm/*/*.csv" \
  --embeddings /scratch1/eibl/data/midterm/embeddings/user_embeddings_minilm.pt \
  --embedding_pool meanpool \
  --history_fraction 0.3 \
  --pseudo_label_margin 2 \
  --out /scratch1/eibl/data/midterm/graphs/retweet_graph.pt
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--max_files` | limit number of CSV files |
| `--max_nodes` | trim the final graph to exactly this many nodes when possible |
| `--history_fraction` | temporal split fraction for `temporal_history` |
| `--future_target_mode` | `new_only` or `all_future` for temporal LP targets |
| `--pseudo_label_margin` | minimum one-sided weak-label evidence count |
| `--keep-isolates / --no-keep-isolates` | keep or drop zero-degree nodes |
| `--no_temporal_views` | skip temporal view construction |

Graph artifact keys:

| Key | Description |
|---|---|
| `x` | node features |
| `edge_index` | directed retweet edges |
| `edge_attr` | edge features |
| `edge_attr_feature_names` | edge feature names |
| `user_ids` | canonical node ids |
| `u2i` | `user_id -> node_index` |
| `feature_names` | node feature names |
| `y` | weak labels, `-1` for unlabeled |
| `label_names` | `["rep", "dem"]` |
| `edge_index_views` | includes `temporal_history` |
| `target_edge_index_views` | includes `temporal_new` |
| `future_edge_index` | LP target edges |
| `data` | compatibility `torch_geometric.data.Data` object |

Node features are 11 graph/account stats plus optional embedding dimensions.

## Step 3: validate and inspect

```bash
python data/data/midterm/scripts/validate_graph.py \
  --graph /scratch1/eibl/data/midterm/graphs/retweet_graph.pt
```

```bash
python data/data/midterm/scripts/inspect_graph.py \
  --graph /scratch1/eibl/data/midterm/graphs/retweet_graph.pt
```

## Training

Example task-1 scripts:

```bash
sbatch scripts/submit_train1_midterm_pl.sh
sbatch scripts/submit_train1_midterm_nm.sh
sbatch scripts/submit_train1_midterm_lp.sh
```

Typical task settings used in recent comparable runs:

| Task | `--task_name` | Typical flags |
|---|---|---|
| Political labels | `classification` | `--n_way 2 --n_shots 3 --n_query 3 --midterm_label_downsample 50:50` |
| Neighbor matching | `neighbor_matching` | `--midterm_edge_view temporal_history --n_way 3 --n_shots 1 --n_query 12` |
| Temporal link prediction | `temporal_link_prediction` | `--midterm_edge_view temporal_history --midterm_target_edge_view temporal_new --n_way 1 --n_shots 1 --n_query 3` |

For embedding-only runs, use:

```text
--midterm_feature_subset emb_only --input_dim 384
```
