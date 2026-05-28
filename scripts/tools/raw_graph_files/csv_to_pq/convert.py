import argparse, glob, json, sys, traceback
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

try:
    import orjson
    def loads(b): return orjson.loads(b)
except ImportError:
    def loads(b): return json.loads(b)

RAW_GLOB = "/scratch1/eibl/data/covid19_twitter/raw/*/*.json"
OUT_DIR  = Path("/scratch1/eibl/data/covid19_twitter/parquet")

SCHEMA = pa.schema([
    ('tweet_id',              pa.int64()),
    ('created_at',            pa.timestamp('s', tz='UTC')),
    ('text',                  pa.string()),
    ('lang',                  pa.string()),
    ('user_id',               pa.int64()),
    ('in_reply_to_status_id', pa.int64()),
    ('in_reply_to_user_id',   pa.int64()),
    ('retweeted_status_id',   pa.int64()),
    ('quoted_status_id',      pa.int64()),
    ('is_quote_status',       pa.bool_()),
    ('mentioned_user_ids',    pa.list_(pa.int64())),
    ('hashtags',              pa.list_(pa.string())),
])

TWITTER_TS_FMT = "%a %b %d %H:%M:%S %z %Y"

def parse_ts(s):
    if not s: return None
    try:
        return int(datetime.strptime(s, TWITTER_TS_FMT).timestamp())
    except (ValueError, TypeError):
        return None

def get_text(tw):
    # For retweets, prefer the original tweet's full text over the truncated "RT @..." prefix
    src = tw.get('retweeted_status') or tw
    et  = src.get('extended_tweet')
    if et and et.get('full_text'):
        return et['full_text']
    return src.get('text') or src.get('full_text')

def extract(tw):
    user     = tw.get('user') or {}
    ents     = tw.get('entities') or {}
    mentions = ents.get('user_mentions') or []
    tags     = ents.get('hashtags') or []
    rt       = tw.get('retweeted_status') or {}
    qt       = tw.get('quoted_status') or {}
    return {
        'tweet_id':              tw.get('id'),
        'created_at':            parse_ts(tw.get('created_at')),
        'text':                  get_text(tw),
        'lang':                  tw.get('lang'),
        'user_id':               user.get('id'),
        'in_reply_to_status_id': tw.get('in_reply_to_status_id'),
        'in_reply_to_user_id':   tw.get('in_reply_to_user_id'),
        'retweeted_status_id':   rt.get('id'),
        'quoted_status_id':      qt.get('id') or tw.get('quoted_status_id'),
        'is_quote_status':       tw.get('is_quote_status'),
        'mentioned_user_ids':    [m['id'] for m in mentions if m.get('id') is not None],
        'hashtags':              [h['text'] for h in tags if h.get('text') is not None],
    }

def flush(buf, writer):
    batch = pa.RecordBatch.from_pydict(buf, schema=SCHEMA)
    writer.write_batch(batch)

def process(files, writer, chunk=50_000):
    buf = {k: [] for k in SCHEMA.names}
    n_ok = n_err = 0
    for fp in files:
        try:
            with open(fp, 'rb') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        rec = extract(loads(line))
                    except Exception:
                        n_err += 1
                        continue
                    for k, v in rec.items():
                        buf[k].append(v)
                    n_ok += 1
                    if n_ok % chunk == 0:
                        flush(buf, writer)
                        buf = {k: [] for k in SCHEMA.names}
        except Exception as e:
            print(f"[ERROR] {fp}: {e}", file=sys.stderr)
            traceback.print_exc()
    if buf['tweet_id']:
        flush(buf, writer)
    return n_ok, n_err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task-id',   type=int, required=True)
    ap.add_argument('--num-tasks', type=int, required=True)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(RAW_GLOB))
    if not files:
        print(f"No files match {RAW_GLOB}", file=sys.stderr); sys.exit(1)

    per   = (len(files) + args.num_tasks - 1) // args.num_tasks
    start = args.task_id * per
    end   = min(start + per, len(files))
    mine  = files[start:end]
    if not mine:
        print(f"Task {args.task_id}: no files (start={start}, total={len(files)})"); return

    out = OUT_DIR / f"covid_{args.task_id:05d}.parquet"
    tmp = out.with_suffix('.parquet.tmp')
    print(f"Task {args.task_id}/{args.num_tasks}: {len(mine)} files [{start}..{end-1}] -> {out}", flush=True)

    with pq.ParquetWriter(tmp, SCHEMA, compression='zstd', compression_level=3) as w:
        n_ok, n_err = process(mine, w)
    tmp.rename(out)
    print(f"Task {args.task_id}: {n_ok:,} tweets, {n_err} errors", flush=True)

if __name__ == '__main__':
    main()