# Ukraine-Russia Twitter Dataset

Retweet graph built from a Twitter dataset collected around the Ukraine-Russia conflict. Nodes are keyed by `userid`. The graph builder can attach text embeddings, build temporal views, and generate weak political labels.

## Raw data

Cluster path:

```text
/project2/ll_774_951/uk_ru/twitter/data/*/*.csv
```

The CSV files use an interleaved format. The loader handles that directly.

## Step 1: build user embeddings

```bash
python data/data/ukr_rus_twitter/scripts/build_user_embeddings.py \
  --csv_glob "/project2/ll_774_951/uk_ru/twitter/data/*/*.csv" \
  --out /scratch1/eibl/data/ukr_rus_twitter/embeddings/user_embeddings_minilm.pt
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--max_files` | limit number of CSV files |
| `--max_nodes` | cap the embedding artifact to exactly this many users when possible |
| `--stop_after_max_nodes / --no-stop_after_max_nodes` | stop at the cap or keep scanning for already-admitted users |
| `--batch_size` | encoder batch size |
| `--max_seq_len` | sentence-transformer max sequence length |

Embedding artifact keys:

| Key | Description |
|---|---|
| `handles` | screen names aligned with the pooled rows |
| `user_ids` | best-effort numeric user ids aligned with `handles` |
| `meanpool` | pooled embedding tensor `[N, D]` |
| `counts` | posts embedded per row |
| `model` | encoder name |

## Step 2: build the graph

```bash
python data/data/ukr_rus_twitter/scripts/generate_retweet_graph.py \
  --csv "/project2/ll_774_951/uk_ru/twitter/data/*/*.csv" \
  --embeddings /scratch1/eibl/data/ukr_rus_twitter/embeddings/user_embeddings_minilm.pt \
  --embedding_pool meanpool \
  --history_fraction 0.3 \
  --pseudo-political-labels \
  --pseudo-label-margin 2 \
  --out /scratch1/eibl/data/ukr_rus_twitter/graphs/retweet_graph_hf03_political_labels.pt
```

Useful flags:

| Flag | Meaning |
|---|---|
| `--max_files` | limit number of CSV files |
| `--max_nodes` | trim the final graph to exactly this many nodes when possible |
| `--history_fraction` | temporal split fraction for `temporal_history` |
| `--future_target_mode` | `new_only` or `all_future` |
| `--pseudo-political-labels` | attach weak labels |
| `--pseudo-label-margin` | minimum one-sided weak-label evidence count |
| `--keep-isolates / --no-keep-isolates` | keep or drop zero-degree nodes |

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
| `label_names` | `["left", "right"]` when labels are attached |
| `edge_index_views` | includes `temporal_history` |
| `target_edge_index_views` | includes `temporal_new` |
| `future_edge_index` | LP target edges |
| `data` | compatibility `torch_geometric.data.Data` object |

## Step 3: validate and inspect

```bash
python data/data/ukr_rus_twitter/scripts/validate_graph.py \
  --graph /scratch1/eibl/data/ukr_rus_twitter/graphs/retweet_graph_hf03_political_labels.pt
```

```bash
python data/data/ukr_rus_twitter/scripts/inspect_graph.py \
  --graph /scratch1/eibl/data/ukr_rus_twitter/graphs/retweet_graph_hf03_political_labels.pt \
  --topk 20
```

## Training

```bash
sbatch scripts/submit_train1_ukr_rus_twitter_pl.sh
sbatch scripts/submit_train1_ukr_rus_twitter_nm.sh
sbatch scripts/submit_train1_ukr_rus_twitter_lp.sh
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
