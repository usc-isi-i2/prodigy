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
- `shards/shard-000000.manifest.json`
- `skipped_rows/shard-000000.skipped.parquet`
- `logs/*.log`

Embeddings are `float16`, 768-dimensional, L2-normalized vectors from `Alibaba-NLP/gte-multilingual-base` at revision `9bbca17d9273fd0d03d5725c7a4b0f6b45142062`.

## Text Assembly

The pipeline creates one canonical content embedding per source row:

- Quote tweet: embed `text` when present, because it captures the user's framing/commentary. A row is treated as a quote when `tweet_type` contains `quote`, `qtd_tweetid` is present, or `qtd_text` is present.
- Retweet: embed `rt_text` when present, because it captures the endorsed content. A row is treated as a retweet when `tweet_type` contains `retweet`, `rt_tweetid` is present, or `text` starts with `RT `.
- Original tweet or reply: embed `text`.
- Fallback order: use `text`, then `rt_text`, then `qtd_text` if the classified source text is absent.

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
cd /dataMeR2/phil/gfm/prodigy
git fetch origin
git switch tweet-embeddings-v001
git pull --ff-only origin tweet-embeddings-v001
git rev-parse HEAD
```

Confirm the checkout is at the current `tweet-embeddings-v001` branch tip before running on tucker.

The embedding manifest records the git commit SHA, command line, package versions, CUDA/Torch runtime, model id/revision, config, source-data fingerprint, and shard checksums.

## Tucker Environment

Use a dedicated conda environment, separate from the existing PRODIGY training environment:

```bash
conda create -n tweet-embeddings-v001 python=3.11 -y
conda activate tweet-embeddings-v001
python -m pip install --upgrade pip
python -m pip install -r scripts/tweet_embeddings/requirements-embeddings.txt
```

Before embedding, confirm the environment sees CUDA:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

The run manifest records the actual package and CUDA versions used.

## GPU Selection

This job defaults to physical GPUs `0,1,2,3` on tucker. Use a different GPU set only if the allocation changes.

CUDA remaps visible GPUs inside Python. With:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
```

the worker log devices map like this:

```text
worker rank 0 device=cuda:0 -> physical GPU 0
worker rank 1 device=cuda:1 -> physical GPU 1
worker rank 2 device=cuda:2 -> physical GPU 2
worker rank 3 device=cuda:3 -> physical GPU 3
```

The `--gpus` argument defaults to `0,1,2,3`, is recorded in config, and also sets `CUDA_VISIBLE_DEVICES`, so keep both aligned in the commands below.

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

Observed gate on tucker passed with these exact counts.

## Tucker Smoke Test

Run a one-shard, one-GPU smoke test before the full run. This writes to a separate smoke output directory. Use `tmux` so the job survives SSH disconnects:

```bash
tmux new -s tweet-embeddings-smoke
conda activate tweet-embeddings-v001
cd /dataMeR2/phil/gfm/prodigy
```

Then run:

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/tweet_embeddings/embed_tweets.py \
  --input-root /dataMeR2/phil/data/ukr_rus_twitter/parquet \
  --output-root /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/gte-multilingual-base/version=v001_smoke \
  --source-files /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/source_files.v001.parquet \
  --gpus 0 \
  --num-workers 1 \
  --smoke-shards 1 \
  --batch-size 2048
```

If the smoke test OOMs, rerun only the smoke test with `--batch-size 1024`.

Monitor smoke logs from another shell:

```bash
tail -f /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/gte-multilingual-base/version=v001_smoke/logs/worker-0.log
```

Validate smoke output:

```bash
python -u scripts/tweet_embeddings/validate_tweet_embeddings.py \
  --output-root /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/gte-multilingual-base/version=v001_smoke \
  --shards 0 \
  --norm-sample 0 \
  --summary-json /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/gte-multilingual-base/version=v001_smoke/validation.json
```

The smoke gate passes when the validator reports shape `[N, 768]`, dtype `float16`, finite values, L2 norms within tolerance, metadata alignment, checksum agreement, and no failed shards.

Observed smoke result on tucker:

- Embedded rows: 498,315
- Skipped rows: 1,685
- Duration: 244.7 seconds
- Throughput: about 2,043 source rows/sec on one GPU
- Smoke validation status: `passed`

## Full Run

After the smoke gate passes, start a long-running tmux session:

```bash
tmux new -s tweet-embeddings
conda activate tweet-embeddings-v001
cd /dataMeR2/phil/gfm/prodigy
```

Then run the 4-GPU job:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u scripts/tweet_embeddings/embed_tweets.py \
  --input-root /dataMeR2/phil/data/ukr_rus_twitter/parquet \
  --output-root /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/gte-multilingual-base/version=v001 \
  --source-files /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/source_files.v001.parquet \
  --gpus 0,1,2,3 \
  --num-workers 4 \
  --batch-size 2048
```

Detach from tmux without stopping the job:

```text
Ctrl-b
d
```

Reconnect later:

```bash
tmux attach -t tweet-embeddings
```

The run is resumable. A shard is skipped only when its embedding file, metadata file, and shard manifest exist; embedding and metadata checksums match; the skipped-row checksum matches when that sidecar exists; embedding shape and dtype match; and metadata row count equals embedding row count.

If a worker OOMs, rerun the same full command with:

```bash
--batch-size 1024
```

## Monitoring

Follow all worker logs:

```bash
tail -f /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/gte-multilingual-base/version=v001/logs/worker-*.log
```

Follow the run-level log:

```bash
tail -f /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/gte-multilingual-base/version=v001/logs/run.log
```

Count completed shards:

```bash
ls /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/gte-multilingual-base/version=v001/shards/*.manifest.json | wc -l
```

Expected full-run shard count:

```text
ceil(220,909,316 / 500,000) = 442
```

Watch GPUs:

```bash
watch -n 2 nvidia-smi
```

The process list should show Python processes on physical GPUs `0,1,2,3`. If a run was accidentally started on the wrong GPUs, stop it with `Ctrl-c` inside tmux. To start the full output fresh, remove only:

```bash
rm -rf /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/gte-multilingual-base/version=v001
```

Do not remove `source_files.v001.parquet`, the staged-Parquet verification JSON, or the smoke output unless you intentionally want to redo those steps.

At the observed smoke speed, the 4-GPU run should take roughly 7.5 to 8 hours if all four GPUs sustain similar throughput.

## Full Validation

Run validation after the full job:

```bash
python -u scripts/tweet_embeddings/validate_tweet_embeddings.py \
  --output-root /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/gte-multilingual-base/version=v001 \
  --norm-sample 10000 \
  --summary-json /dataMeR2/phil/data/ukr_rus_twitter/tweet_embeddings/gte-multilingual-base/version=v001/validation.json
```

For a slower complete norm check, use `--norm-sample 0`.
