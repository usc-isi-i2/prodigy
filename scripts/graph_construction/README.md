# Graph Construction

This directory contains graph builders for staged social graph artifacts.

## Ukraine/Russia Retweet Graph Builder

Script:
`/Users/philipp/projects/gfm/prodigy/scripts/graph_construction/generate_retweet_graph_from_parquet.py`

### What It Builds

This script builds a directed Twitter retweet graph for the Ukraine/Russia parquet corpus.

Edge semantics:
- `A -> B` means user `A` retweeted user `B`
- edge weight is raw `n_retweets`
- one edge is emitted per `(userid, rt_userid)` pair

Node features:
- node features come from the bio embedding store
- embeddings are resolved through:
  - `user_bio_observations.parquet`
  - `bio_embedding_index.parquet`
  - shard `.npy` files

Bio selection policy:
- default with no cutoff: latest observed bio overall
- with `--graph-cutoff`: latest observed bio at or before the cutoff
- users without a resolved embedding get a zero vector

### Required Inputs

#### 1. Staged tweet parquet files

By default the script reads from:

```text
/dataMeR2/phil/data/ukr_rus_twitter/parquet
```

It expects at least these parquet columns:

```text
userid
rt_userid
rt_screen
date
description
rt_user_description
```

The graph construction itself uses:

```text
userid
rt_userid
rt_screen
date
```

#### 2. Bio embedding store

By default the script reads from:

```text
/dataMeR2/phil/data/ukr_rus_twitter/bio_embeddings/gte-multilingual-base/version=v001
```

It expects these artifacts to exist under that root:

```text
user_bio_observations.parquet
bio_embedding_index.parquet
shards/*.npy
```

### Outputs

By default the script writes:

```text
data/data/ukr_rus_twitter/graphs/retweet_graph_parquet.pt
data/data/ukr_rus_twitter/graphs/retweet_graph_parquet.meta.json
```

The `.pt` artifact preserves the repo’s current graph contract and includes at least:

```text
data
user_ids
u2i
feature_names
edge_attr
edge_attr_feature_names
edge_index
x
y
```

Important output properties:
- `user_ids` are sorted and stable
- `len(user_ids) == data.x.shape[0]`
- `edge_attr_feature_names = ["n_retweets"]`
- `edge_attr[:, 0]` is raw retweet count, not log-transformed

The `.meta.json` file records counts, embedding stats, cutoff info, and temporal-view stats.

### Cleaning Rules

The script:
- drops rows with missing `userid`
- drops rows with missing `rt_userid`
- drops self-retweets where `userid == rt_userid`
- parses `date` into a timestamp
- drops invalid-date rows unless `--strict-dates` is set

### Temporal Views

Unless `--no-temporal-views` is passed, the script also writes temporal edge views:

- `edge_index_views["retweet_all"]`
- `edge_index_views["temporal_history"]`
- `target_edge_index_views["temporal_new"]`
- `future_edge_index`

History/future splitting is controlled by:
- `--history-fraction`
- `--future-target-mode`

### Progress Logging

The script prints `[progress]` messages for:
- parquet discovery
- DuckDB source scan
- date validation
- retweet event materialization
- edge aggregation
- handle resolution
- bio embedding resolution
- per-shard embedding loads
- temporal view construction
- artifact validation
- graph and metadata writes

### How To Run

From the repo root:

```bash
cd /Users/philipp/projects/gfm/prodigy
python scripts/graph_construction/generate_retweet_graph_from_parquet.py
```

Run with explicit input and output roots:

```bash
python scripts/graph_construction/generate_retweet_graph_from_parquet.py \
  --parquet-root /dataMeR2/phil/data/ukr_rus_twitter/parquet \
  --bio-embeddings-root /dataMeR2/phil/data/ukr_rus_twitter/bio_embeddings/gte-multilingual-base/version=v001 \
  --out data/data/ukr_rus_twitter/graphs/retweet_graph_parquet.pt
```

Run a small smoke test:

```bash
python scripts/graph_construction/generate_retweet_graph_from_parquet.py \
  --max-files 10 \
  --out /tmp/retweet_graph_parquet_smoke.pt
```

Build a cutoff graph:

```bash
python scripts/graph_construction/generate_retweet_graph_from_parquet.py \
  --graph-cutoff "2022-06-01 00:00:00" \
  --out data/data/ukr_rus_twitter/graphs/retweet_graph_parquet_2022-06-01.pt
```

Build with explicit DuckDB settings:

```bash
python scripts/graph_construction/generate_retweet_graph_from_parquet.py \
  --duckdb-threads 32 \
  --duckdb-memory-limit 200GB \
  --duckdb-temp-dir /tmp/duckdb_ukr_rus_graph
```

### Main CLI Arguments

- `--parquet-root`: recursive root for parquet discovery
- `--parquet-path`: explicit parquet file or directory; repeatable
- `--bio-embeddings-root`: root containing bio embedding artifacts
- `--out`: output `.pt` path
- `--graph-cutoff`: optional inclusive timestamp cutoff
- `--max-files`: limit parquet files for smoke tests
- `--strict-dates`: fail instead of dropping invalid-date rows
- `--history-fraction`: split point for temporal history/future views
- `--future-target-mode`: `new_only` or `all_future`
- `--no-temporal-views`: disable temporal view generation
- `--duckdb-memory-limit`: optional DuckDB memory cap
- `--duckdb-threads`: optional DuckDB thread count
- `--duckdb-temp-dir`: optional DuckDB temp directory

### Environment

The script requires:
- `duckdb`
- `pyarrow`
- `numpy`
- `torch`
- `torch-geometric`

If `duckdb` is missing, the script fails immediately with an explicit error.

### Notes

- This builder uses parquet directly and does not use the legacy CSV pipeline.
- The primary edge weight is always raw `n_retweets`.
- Handle metadata is best-effort and comes from `rt_screen`.
