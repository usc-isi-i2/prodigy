# CP-HK Twitter Retweet Graph With Bio Features

This document tracks the COSINE CP-HK Twitter graph build and NM training run.

## Dataset

Source: CARC `/project2/emiliofe_74/data_backup/COSINE/2022CP-HK`.

Chosen files:

```text
an_cp-hk.twitter.v7-ground-truth.2020-04-07_2020-08-23.json.gz
an_cp-hk.twitter.v7-ground-truth.2020-08-24_2020-09-13.json.gz
```

These are non-Russia COSINE Twitter JSONL gzip files. Retweet edges are derived
from `user.id_h -> retweeted_status.user.id_h`. Node bios use
`user.description_m` and `retweeted_status.user.description_m`.

## Tucker Paths

This run uses `/dataMeR1` by request:

```text
/dataMeR1/phil/data/cp_hk_twitter/raw
/dataMeR1/phil/data/cp_hk_twitter/parquet
/dataMeR1/phil/data/cp_hk_twitter/embeddings
/dataMeR1/phil/data/cp_hk_twitter/graphs
/dataMeR1/phil/gfm/prodigy
/dataMeR1/phil/logs
```

## Environments

Build/parquet/embedding/graph generation:

```bash
source /home/mhchu/miniconda3/etc/profile.d/conda.sh
conda activate tweet-embeddings-v001
```

Training:

```bash
source /home/mhchu/miniconda3/etc/profile.d/conda.sh
conda activate prodigy
```

## Commands

Transfer raw data from CARC to Tucker:

```bash
bash scripts/experiments/setup/cp_hk_twitter/transfer_cp_hk_raw_to_tucker.sh
```

The end-to-end Tucker command is:

```bash
cd /dataMeR1/phil/gfm/prodigy
GPU_ID=1 nohup bash scripts/experiments/setup/cp_hk_twitter/run_cp_hk_nm_tucker.sh \
  > /dataMeR1/phil/logs/cp_hk_nm_$(date +%Y%m%d_%H%M%S).out \
  2> /dataMeR1/phil/logs/cp_hk_nm_$(date +%Y%m%d_%H%M%S).err &
```

For smoke tests:

```bash
MAX_RECORDS=100000 BUILD_ONLY=1 bash scripts/experiments/setup/cp_hk_twitter/run_cp_hk_nm_tucker.sh
```

## Artifacts

Expected graph:

```text
/dataMeR1/phil/data/cp_hk_twitter/graphs/retweet_graph.pt
```

The graph stores:

- `x`: 768-dim GTE bio embeddings.
- `edge_index`: all directed retweet edges.
- `edge_attr`: `log1p(n_retweets)`.
- `edge_index_views["temporal_history"]`: history slice.
- `target_edge_index_views["future"]` and `["temporal_new"]`: future new edges.
- `user_ids`: COSINE hashed user ids.

## Status

Pending: raw transfer, full build, and NM training launch.
