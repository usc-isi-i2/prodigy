# TwiBot-20

TwiBot-20 is a Twitter bot-detection benchmark (Feng et al., *TwiBot-20: A
Comprehensive Twitter Bot Detection Benchmark*, CIKM 2021). We use it as an
additional social-graph node-classification task (bot vs. human) for
cross-task / cross-dataset transfer experiments, alongside `ukr_rus_twitter`,
`covid19_twitter`, and `midterm`.

The dataset is **access-gated** (request from the authors; do not redistribute
the raw data, per Twitter/X ToS). The raw files below are **not** committed to
this repo — they live only on the cluster.

## Contents

| Path | Purpose |
| --- | --- |
| `scripts/node_json_to_bio_parquet.py` | Convert `node.json` → bio-embedding-ready Parquet (user nodes only). |
| `../../../scripts/bio_embeddings/configs/twibot20_index_smoke.yaml` | Index-only smoke config for the bio pipeline. |
| `../../../scripts/bio_embeddings/configs/twibot20_full.yaml` | Full bio-embedding run config. |

## Source layout (Format22)

Downloaded as `TwiBot-20-Format22-001.zip` (2.52 GB compressed, ~7.4 GB
extracted). This is the TwiBot-22-schema packaging of TwiBot-20. Three other
zips exist (`Twibot-20.zip` original format, `Twibot20_processed_data.zip`,
`data.zip`) — alternate packagings we do not use here.

On tucker: `/dataMeR1/phil/data/twibot20/raw/Twibot-20/`

| File | Size | Content |
| --- | --- | --- |
| `node.json` | 5.77 GB | Single JSON array of all nodes. User nodes have `id` = `u…` + `description`, `public_metrics`, `verified`, `location`, `created_at`. Tweet nodes are `t…`. |
| `edge.csv` | 952 MB | `source_id, relation, target_id` — relations: `follow`, `friend`, `post` (user→tweet). |
| `label.csv` | 220 KB | `id,label` → `bot` / `human`. |
| `split.csv` | 5 MB | `id,split` → `train` / `val` / `test` / `support`. |
| `user_info.pt` | 717 MB | Preprocessed user tensor (unused by this pipeline). |

### Measured stats (full scan of `node.json`)

- 33,717,772 total nodes = **229,580 users** + 33,488,192 tweets.
- Users with a non-empty bio: **187,731 (81.8%)**, avg 89.5 chars.
- Labels: 11,826 labeled users — **6,589 bot / 5,237 human**.
- Splits: 8,278 train / 2,365 val / 1,183 test / 217,754 support (`support` =
  unlabeled neighbor pool).
- Edges: `follow` 110,869 · `friend` 117,110 · `post` 33,488,192.

## Transfer & extract (from a laptop with the download)

```bash
ssh tucker 'mkdir -p /dataMeR1/phil/data/twibot20/raw'
rsync -avP --partial ~/Downloads/TwiBot-20-Format22-001.zip \
  tucker:/dataMeR1/phil/data/twibot20/raw/
ssh tucker 'cd /dataMeR1/phil/data/twibot20/raw && unzip -o TwiBot-20-Format22-001.zip'
```

## Step 1 — Convert `node.json` → Parquet

The bio pipeline reads Parquet, not JSON. This step streams `node.json` with
`ijson` (constant memory), keeps only user nodes, and writes `userid`,
`description`, `created_at` for **all ~230k user nodes**. The `u…` id prefix is
preserved so the output joins to `label.csv` / `split.csv` / `edge.csv`.

```bash
# conda is not on PATH for non-login ssh; source it first.
source /home/mhchu/miniconda3/etc/profile.d/conda.sh
conda activate bio-embeddings-v001   # has pyarrow; `pip install ijson` if missing

python data/data/twibot20/scripts/node_json_to_bio_parquet.py \
  --input  /dataMeR1/phil/data/twibot20/raw/Twibot-20/node.json \
  --output /dataMeR1/phil/data/twibot20/parquet/users/users-000.parquet
```

Output: `/dataMeR1/phil/data/twibot20/parquet/` — the `input_root` the bio
pipeline globs (`**/*.parquet`). Expected: ~230k rows, well under 100 MB.

**Why no pipeline change is needed:** the indexer
(`scripts/bio_embeddings/indexer.py`) auto-detects the user-id and description
columns via `_coalesce_varchar(["userid", …])` / `["description", …]`, parses
the Twitter `created_at` format in `_timestamp_expr`, and degrades absent
retweet/quote bio columns to `NULL`/`FALSE`. TwiBot-20 therefore yields only
`author`-role bio observations.

## Step 2 — Index-only smoke

```bash
cd /dataMeR1/phil/gfm/prodigy
python -u scripts/bio_embeddings/embed_bios.py \
  --config scripts/bio_embeddings/configs/twibot20_index_smoke.yaml
```

Check `…/version=v001_smoke/bio_index_summary.json` — distinct bio count should
be ~180k, all source role `author`.

## Step 3 — Full embedding run

Small enough for a single GPU (~188k bios → minutes). Run on a free GPU
(`nvidia-smi` first):

```bash
cd /dataMeR1/phil/gfm/prodigy
CUDA_VISIBLE_DEVICES=0 python -u scripts/bio_embeddings/embed_bios.py \
  --config scripts/bio_embeddings/configs/twibot20_full.yaml
```

Validate:

```bash
python -u scripts/bio_embeddings/validate_bio_embeddings.py \
  --output-root /dataMeR1/phil/data/twibot20/bio_embeddings/gte-multilingual-base/version=v001 \
  --summary-json /dataMeR1/phil/data/twibot20/bio_embeddings/gte-multilingual-base/version=v001/validation.json
```

Embeddings are fp16, 768-d, L2-normalized from `Alibaba-NLP/gte-multilingual-base`
(revision `9bbca17d9273fd0d03d5725c7a4b0f6b45142062`), keyed by `bio_hash`. See
`scripts/bio_embeddings/README.md` for the full artifact contract.

## Retweet graph

We build the task graph as a **retweet** graph (not the native follow/friend
graph), preserving the bot/human labels on user nodes.

Why retweet: the native `follow`/`friend` edges (`edge.csv`) form a sparse
ego-net star — only the ~11.9k annotated users have out-edges (227,979 edges
total; neighbors are leaves). Reconstructed retweets give ~2.0M directed
user→user edges with genuine multi-hop connectivity. Trade-off: the ego-net
guarantees every labeled user is connected, whereas ~12.8% of labeled users are
isolated in the retweet graph (983 bots / 534 humans).

Measured retweet graph: **2,010,925 distinct directed edges** (retweeter→
retweeted, in-set, self-loops dropped), **164,959 participant users**.

### Step 4 — Extract retweet edges

TwiBot-20 has no native retweet edges and tweet nodes are only `{id, text}`, so
edges are reconstructed: `user --post--> tweet(text "RT @handle:")`, then
`handle → rt_userid` via the username table. Intermediates live under
`graph_build/` (kept out of `parquet/` so the bio pipeline never ingests them).

```bash
source /home/mhchu/miniconda3/etc/profile.d/conda.sh && conda activate bio-embeddings-v001
cd /dataMeR1/phil/gfm/prodigy
python -u data/data/twibot20/scripts/extract_retweet_edges.py \
  --node-json /dataMeR1/phil/data/twibot20/raw/Twibot-20/node.json \
  --edge-csv  /dataMeR1/phil/data/twibot20/raw/Twibot-20/edge.csv \
  --out       /dataMeR1/phil/data/twibot20/graph_build/retweet_edges.parquet
```

Output: `graph_build/retweet_edges.parquet` (`userid`, `rt_userid`, `n_retweets`).

### Step 5 — Build the graph

Nodes = retweet-edge participants ∪ all labeled users. Features = bio embeddings,
**zero-filled** for users without a bio (matches the covid/ukr_rus convention).
`y`: `human=0`, `bot=1`, unlabeled `support=-1`. Train/val/test/support stored as
node masks (`data.{train,val,test,support}_mask`).

```bash
python -u scripts/graph_construction/generate_twibot20_retweet_graph.py \
  --edges  /dataMeR1/phil/data/twibot20/graph_build/retweet_edges.parquet \
  --bio-embeddings-root /dataMeR1/phil/data/twibot20/bio_embeddings/gte-multilingual-base/version=v001 \
  --out    /dataMeR1/phil/data/twibot20/graphs/retweet_graph.pt
```

Output: `graphs/retweet_graph.pt` + `graphs/retweet_graph.meta.json`. The
artifact schema matches the other retweet graphs (`x`, `edge_index`, `edge_attr`,
`y`, `user_ids`, `feature_names`, `label_names`, plus a `data` PyG object). Not
yet wired into the PRODIGY dataloader — that is a separate step.
