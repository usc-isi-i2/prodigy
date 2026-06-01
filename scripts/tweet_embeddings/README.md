# Tweet Embedding Pipeline

This directory contains the reproducible tweet-level embedding pipeline for the Ukraine-Russia Twitter Parquet corpus on tucker.

## What It Produces

The full run writes deterministic shard artifacts to:

```text
/dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/gte-multilingual-base/version=v001/
```

Outputs include:

- `config.yaml`
- `manifest.json`
- `manifest.parquet`
- `source_files.parquet`
- `shards/shard-000000.emb.npy`
- `shards/shard-000000.meta.parquet`
- `skipped_rows/shard-000000.skipped.parquet`
- `logs/*.log`

Embeddings are `float16`, 768-dimensional, L2-normalized vectors from `Alibaba-NLP/gte-multilingual-base` at revision `9bbca17d9273fd0d03d5725c7a4b0f6b45142062`.

## Text Assembly

The pipeline creates one canonical content embedding per source row:

- Pure retweet: embed `rt_text`, because it captures the endorsed content.
- Quote tweet: embed `text`, because it captures the user's framing/commentary.
- Original tweet or reply: embed `text`.

Before tokenization it applies Unicode NFKC normalization, replaces URLs with `<URL>`, replaces handles with `<USER>`, collapses whitespace, and preserves case, punctuation, hashtags, emojis, and multilingual characters.

Rows are skipped, not silently dropped, when the assembled text is empty after preprocessing or matches deleted/unavailable/withheld placeholder text. Skipped rows are written to the shard-aligned skipped sidecar.

## Branch And Commit

Local branch:

```bash
cd /Users/philippeibl/projects/gfm/prodigy
git switch tweet-embeddings-v001
git status --short
git push -u origin tweet-embeddings-v001
```

On tucker:

```bash
cd /home1/eibl/gfm/prodigy
git fetch origin
git switch tweet-embeddings-v001
git pull --ff-only origin tweet-embeddings-v001
git rev-parse HEAD
```

The embedding manifest records the git commit SHA, command line, package versions, CUDA/Torch runtime, model id/revision, config, source-data fingerprint, and shard checksums.

## Tucker Environment

Use a dedicated environment, separate from the existing PRODIGY training environment:

```bash
python3 -m venv /dataMeR2/phil/envs/tweet-embeddings-v001
source /dataMeR2/phil/envs/tweet-embeddings-v001/bin/activate
python -m pip install --upgrade pip
python -m pip install -r scripts/tweet_embeddings/requirements-embeddings.txt
```

If tucker already has a preferred CUDA/PyTorch module, load that first and then install the remaining requirements. The run manifest will record the actual package and CUDA versions used.

## Verify Staged Parquet

The staged local mirror must pass verification before embedding:

```bash
mkdir -p /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings

python -u scripts/tweet_embeddings/verify_staged_parquet.py \
  --input-root /dataMeR2/phil/data/ukr_rus_twitter/parquet \
  --checksum \
  --output-source-files /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/source_files.v001.parquet \
  --summary-json /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/verify_staged_parquet.v001.json
```

Expected gate:

- `2022-02`: 158 files, 27,830,486 rows
- `2022-03`: 719 files, 121,461,576 rows
- `2022-04`: 621 files, 71,617,254 rows
- Total: 1,498 files, 220,909,316 rows

## Tucker Smoke Test

Run a one-shard, one-GPU smoke test before the full run. This writes to a separate smoke output directory:

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/tweet_embeddings/embed_tweets.py \
  --input-root /dataMeR2/phil/data/ukr_rus_twitter/parquet \
  --output-root /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/gte-multilingual-base/version=v001_smoke \
  --source-files /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/source_files.v001.parquet \
  --gpus 1 \
  --num-workers 1 \
  --smoke-shards 1 \
  --batch-size 2048
```

If the smoke test OOMs, rerun only the smoke test with `--batch-size 1024`.

Validate smoke output:

```bash
python -u scripts/tweet_embeddings/validate_tweet_embeddings.py \
  --output-root /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/gte-multilingual-base/version=v001_smoke \
  --shards 0 \
  --norm-sample 0 \
  --summary-json /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/gte-multilingual-base/version=v001_smoke/validation.json
```

The smoke gate passes when the validator reports shape `[N, 768]`, dtype `float16`, finite values, L2 norms within tolerance, metadata alignment, checksum agreement, and no failed shards.

## Full Run

After the smoke gate passes:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 python -u scripts/tweet_embeddings/embed_tweets.py \
  --input-root /dataMeR2/phil/data/ukr_rus_twitter/parquet \
  --output-root /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/gte-multilingual-base/version=v001 \
  --source-files /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/source_files.v001.parquet \
  --gpus 1,2,3,4 \
  --num-workers 4 \
  --batch-size 2048
```

The run is resumable. A shard is skipped only when its embedding file, metadata file, manifest, shape, dtype, row counts, and checksums validate.

If a worker OOMs, rerun the same full command with:

```bash
--batch-size 1024
```

## Full Validation

Run validation after the full job:

```bash
python -u scripts/tweet_embeddings/validate_tweet_embeddings.py \
  --output-root /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/gte-multilingual-base/version=v001 \
  --norm-sample 10000 \
  --summary-json /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/gte-multilingual-base/version=v001/validation.json
```

For a slower complete norm check, use `--norm-sample 0`.
