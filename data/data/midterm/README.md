# Midterm Twitter Dataset

Retweet graph built from the 2022 US midterm election Twitter dataset. Nodes are users, edges are retweet relationships. Node labels are political leaning (Democrat / Republican) inferred from hashtags.

## Raw data

CSV files organised as `<dataset_root>/*/*.csv`. On the cluster the data lives at:

```
/project2/ll_774_951/midterm/*/*.csv
```

Each CSV row is one tweet and must contain at minimum: `userid`, `screen_name`, `rt_userid`, `rt_screen`, `date`, `followers_count`, `verified`, `statuses_count`.

## Pipeline

Run the three steps below in order. The midterm dataset also has a convenience script [`scripts/run_pipeline.sh`](scripts/run_pipeline.sh) that chains all three steps.

### Step 1 — Build user text embeddings

Encodes each user's tweet/bio text with a sentence transformer and mean-pools them into a single vector per user.

```bash
python data/data/midterm/scripts/build_user_embeddings.py \
  --csv_glob "/project2/ll_774_951/midterm/*/*.csv" \
  --model "sentence-transformers/all-MiniLM-L6-v2" \
  --out data/data/midterm/embeddings/embeddings_all-MiniLM-L6-v2.pt
```

| Argument | Default | Description |
|---|---|---|
| `--csv_glob` | `/project2/ll_774_951/midterm/*/*.csv` | Glob for raw CSV files |
| `--model` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace sentence-transformer model |
| `--out` | `data/data/midterm/embeddings/embeddings_<model>.pt` | Output path |
| `--batch_size` | 256 | Encoding batch size |
| `--max_files` | 0 (all) | Limit number of CSV files (useful for testing) |
| `--device` | auto | `cuda` or `cpu` |

**Output:** a `.pt` file containing a dict with keys `user_ids` (list of user ID strings) and `meanpool` (tensor of shape `[num_users, emb_dim]`).

### Step 2 — Build the retweet graph

Constructs a PyG `Data` object from the raw CSVs. Optionally attaches the text embeddings from Step 1.

```bash
python data/data/midterm/scripts/build_retweet_graph.py \
  --csv_glob "/project2/ll_774_951/midterm/*/*.csv" \
  --embeddings data/data/midterm/embeddings/embeddings_all-MiniLM-L6-v2.pt \
  --embedding_pool meanpool \
  --out data/data/midterm/graphs/retweet_graph.pt
```

| Argument | Default | Description |
|---|---|---|
| `--csv_glob` | `/project2/ll_774_951/midterm/*/*.csv` | Glob for raw CSV files |
| `--out` | `data/data/midterm/graphs/retweet_graph.pt` | Output path |
| `--embeddings` | `` (none) | Path to embeddings `.pt` from Step 1 |
| `--embedding_pool` | `meanpool` | Which pool to use: `meanpool` or `maxpool` |
| `--history_fraction` | 0.8 | Fraction of edges used as "history" for temporal LP |
| `--label_source` | `political_leaning` | Node labels: `political_leaning` (hashtag-based R/D) or `state` |
| `--future_target_mode` | `new_only` | LP target edges: `new_only` (unseen pairs) or `all_future` |
| `--no_temporal_views` | off | Skip building temporal train/val/test edge splits |
| `--max_files` | 0 (all) | Limit number of CSV files |

**Output:** a `.pt` dict with keys:

| Key | Shape | Description |
|---|---|---|
| `x` | `[N, F]` | Node features (11 numeric + optional 384-dim embeddings) |
| `edge_index` | `[2, E]` | Retweet edges |
| `edge_attr` | `[E, 4]` | Edge features: `first_retweet_time`, `n_retweets`, `avg_rt_fav`, `avg_rt_reply` |
| `y` | `[N]` | Node labels |
| `user_ids` | `[N]` | User ID strings |
| `feature_names` | list | Names of the 11 numeric node features |
| `label_names` | list | Label class names |
| `temporal_*` | — | Temporal train/val/test edge masks (if `--no_temporal_views` is not set) |

Node features (11): `subscriber_count`, `verified`, `avg_favorites`, `avg_comments`, `avg_score`, `avg_n_hashtags`, `avg_n_mentions`, `avg_has_media`, `post_count`, `in_degree`, `out_degree`.

### Step 3 — Validate

```bash
python data/data/midterm/scripts/validate_graph.py \
  --graph data/data/midterm/graphs/retweet_graph.pt
```

Prints a summary and exits non-zero if required keys are missing or shapes are inconsistent.

## Running on the cluster

Submit the SLURM scripts from the repo root. Update the paths in `--out` / `--root` to your scratch directory.

```bash
# Embeddings (GPU, ~30 min)
sbatch data/data/midterm/scripts/run_pipeline.sh

# Or run the full pipeline as a shell script
bash data/data/midterm/scripts/run_pipeline.sh
```

The experiment training scripts are in [`scripts/`](../../../scripts/):

```bash
sbatch scripts/submit_train1_midterm_lp.sh   # temporal link prediction
sbatch scripts/submit_train1_midterm_nm.sh   # neighbor matching
sbatch scripts/submit_train1_midterm_pl.sh   # political leaning classification
```

## Experiments

Pass `--root <dir>` pointing to the folder that contains your graph `.pt` file and `--graph_filename <file>`.

| Task | `--task_name` | Key flags |
|---|---|---|
| Temporal link prediction | `temporal_link_prediction` | `--n_way 1 --n_shots 1 --n_query 3` |
| Neighbor matching | `neighbor_matching` | `--n_way 3 --n_shots 3 --n_query 24` |
| Political leaning classification | `classification` | `--n_way 2 --n_shots 4 --n_query 3 --midterm_label_downsample 50:50` |

Set `--midterm_feature_subset emb_only` and `--input_dim 384` when using text embeddings only.
