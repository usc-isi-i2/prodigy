# Bio Embedding Pipeline

This directory contains the reproducible bio-level embedding pipeline for staged Twitter Parquet corpora on tucker.

The pipeline stores one embedding per distinct normalized bio text, keyed by `bio_hash`, and stores user/time/source-role provenance separately. It does not modify or depend on existing tweet embedding outputs.

Supported source layouts today:

- flat `ukr_rus_twitter` parquet with columns such as `description`, `rt_user_description`, and `qtd_user_description`
- nested `covid19_twitter/parquet/raw_nested` parquet with Twitter JSON structs such as `user.description`, `retweeted_status.user.description`, and `quoted_status.user.description`
- deduplicated Facebook page-profile parquet with `account_id`, `page_description`, and `metadata_observed_at`

## What It Produces

The full run writes deterministic artifacts to:

```text
/dataMeR1/phil/data/ukr_rus_twitter/bio_embeddings/gte-multilingual-base/version=v001/
```

Core outputs include:

- `config.yaml`
- `manifest.json`
- `manifest.parquet`
- `source_files.parquet`
- `bio_index_summary.json`
- `bio_texts.parquet`
- `user_bio_observations.parquet`
- `bio_embedding_index.parquet`
- `shards/shard-000000.emb.npy`
- `shards/shard-000000.meta.parquet`
- `shards/shard-000000.manifest.json`
- `logs/*.log`

Embeddings are `float16`, 768-dimensional, L2-normalized vectors from `Alibaba-NLP/gte-multilingual-base` at revision `9bbca17d9273fd0d03d5725c7a4b0f6b45142062`.

## Bio Extraction

The indexer scans the staged Parquet corpus with DuckDB using projected columns and `read_parquet(..., filename=true, file_row_number=true)`. Current staged source metadata on tucker is:

```text
files: 1,498
rows:  220,909,316
```

Source roles:

- `author`: `userid`, `description`
- `retweeted_author`: `rt_userid`, `rt_user_description`
- `quoted_author`: `qtd_userid`, `qtd_user_description` on non-retweet-like rows
- `retweeted_quoted_author`: `qtd_userid`, `qtd_user_description` on retweet-like rows

Retweet-like follows the tweet embedding pipeline spirit: `tweet_type` contains retweet, `rt_tweetid` is present, `rt_text` is present, or `text` starts with `RT `.

## Normalization

The bio text policy is `bio-text-v001`:

- Unicode NFKC
- URLs become `<URL>`
- handles become `<USER>`
- whitespace is collapsed
- leading/trailing whitespace is trimmed

The embedding key is SHA-256 over the normalized bio text. Source role is provenance only.

## Artifact Contract

`bio_texts.parquet` has one row per distinct normalized bio:

```text
bio_id, bio_hash, normalized_bio_text, n_observations, first_seen_at, last_seen_at
```

`user_bio_observations.parquet` has one row per `(userid, bio_hash)`:

```text
userid, bio_hash, first_seen_at, last_seen_at, n_observations,
n_author_observations, n_retweeted_author_observations,
n_quoted_author_observations, n_retweeted_quoted_author_observations,
source_roles, first_tweetid, last_tweetid,
first_global_row_id, last_global_row_id,
first_source_file, last_source_file,
first_source_file_index, last_source_file_index,
first_source_offset, last_source_offset
```

`source_roles` is a comma-separated canonical role list in role-order.

`bio_embedding_index.parquet` resolves vectors:

```text
bio_id, bio_hash, embedding_shard, embedding_row,
embedding_dim, embedding_dtype, model, revision
```

Graph code should resolve features as:

```text
user node + cutoff/window -> bio_hash(es) from user_bio_observations
bio_hash -> bio_embedding_index -> embedding shard row
```

No latest/first/mean/window policy is baked into this base artifact.

## Tucker Environment

Use a dedicated conda environment:

```bash
conda create -n bio-embeddings-v001 python=3.11 -y
conda activate bio-embeddings-v001
python -m pip install --upgrade pip
python -m pip install -r scripts/bio_embeddings/requirements-embeddings.txt
```

Confirm CUDA:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

## Index Smoke

Run an index-only smoke on a separate output root first:

```bash
tmux new -s bio-embeddings-smoke
conda activate bio-embeddings-v001
cd /dataMeR1/phil/gfm/prodigy
```

```bash
python -u scripts/bio_embeddings/embed_bios.py \
  --input-root /dataMeR1/phil/data/ukr_rus_twitter/parquet \
  --output-root /dataMeR1/phil/data/ukr_rus_twitter/bio_embeddings/gte-multilingual-base/version=v001_smoke \
  --source-files /dataMeR1/phil/data/ukr_rus_twitter/tweet_embeddings/source_files.v001.parquet \
  --index-only \
  --rebuild-bio-index \
  --no-source-checksums \
  --duckdb-memory-limit 200GB \
  --duckdb-threads 32 \
  --keep-work-dir
```

For a faster functional check on a tiny local fixture, run the tests rather than the full corpus index.

## One-Shard Embedding Smoke

After the smoke index succeeds:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/bio_embeddings/embed_bios.py \
  --input-root /dataMeR1/phil/data/ukr_rus_twitter/parquet \
  --output-root /dataMeR1/phil/data/ukr_rus_twitter/bio_embeddings/gte-multilingual-base/version=v001_smoke \
  --source-files /dataMeR1/phil/data/ukr_rus_twitter/tweet_embeddings/source_files.v001.parquet \
  --gpus 0 \
  --num-workers 1 \
  --smoke-shards 1 \
  --batch-size 2048
```

Validate:

```bash
python -u scripts/bio_embeddings/validate_bio_embeddings.py \
  --output-root /dataMeR1/phil/data/ukr_rus_twitter/bio_embeddings/gte-multilingual-base/version=v001_smoke \
  --shards 0 \
  --norm-sample 0 \
  --summary-json /dataMeR1/phil/data/ukr_rus_twitter/bio_embeddings/gte-multilingual-base/version=v001_smoke/validation.json
```

If the smoke embedding OOMs, rerun with `--batch-size 1024`.

## Full Run

After the smoke gate passes:

```bash
tmux new -s bio-embeddings
conda activate bio-embeddings-v001
cd /dataMeR1/phil/gfm/prodigy
```

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u scripts/bio_embeddings/embed_bios.py \
  --input-root /dataMeR1/phil/data/ukr_rus_twitter/parquet \
  --output-root /dataMeR1/phil/data/ukr_rus_twitter/bio_embeddings/gte-multilingual-base/version=v001 \
  --source-files /dataMeR1/phil/data/ukr_rus_twitter/tweet_embeddings/source_files.v001.parquet \
  --gpus 0,1,2,3 \
  --num-workers 4 \
  --batch-size 2048 \
  --duckdb-memory-limit 200GB \
  --duckdb-threads 32
```

The run is resumable. A shard is skipped only when the embedding file, metadata file, and shard manifest exist; checksums match; shape and dtype match; and metadata row count equals embedding row count.

Validate the completed store:

```bash
python -u scripts/bio_embeddings/validate_bio_embeddings.py \
  --output-root /dataMeR1/phil/data/ukr_rus_twitter/bio_embeddings/gte-multilingual-base/version=v001 \
  --summary-json /dataMeR1/phil/data/ukr_rus_twitter/bio_embeddings/gte-multilingual-base/version=v001/validation.json
```

## Monitoring

Follow worker logs:

```bash
tail -f /dataMeR1/phil/data/ukr_rus_twitter/bio_embeddings/gte-multilingual-base/version=v001/logs/worker-*.log
```

Follow the run log:

```bash
tail -f /dataMeR1/phil/data/ukr_rus_twitter/bio_embeddings/gte-multilingual-base/version=v001/logs/run.log
```

Count completed shards:

```bash
ls /dataMeR1/phil/data/ukr_rus_twitter/bio_embeddings/gte-multilingual-base/version=v001/shards/*.manifest.json | wc -l
```

Watch GPUs:

```bash
watch -n 2 nvidia-smi
```

## COVID-19 Twitter Run

For the nested COVID parquet mirror, point the same pipeline at `raw_nested` and a COVID output root:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u scripts/bio_embeddings/embed_bios.py \
  --input-root /dataMeR1/phil/data/covid19_twitter/parquet/raw_nested \
  --output-root /dataMeR1/phil/data/covid19_twitter/bio_embeddings/gte-multilingual-base/version=v001 \
  --gpus 0,1,2,3 \
  --num-workers 4 \
  --batch-size 2048 \
  --duckdb-memory-limit 200GB \
  --duckdb-threads 32
```

If you already have a verified `source_files.parquet` for the COVID parquet mirror, pass it with `--source-files` the same way as the Ukraine/Russia run.
